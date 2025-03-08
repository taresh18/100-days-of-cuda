import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    stage: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if stage == 1:
        # from 0 to the left of the diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif stage == 2:
        # used only for block in which there is transition between masked and non-masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        # for compiler optimisation
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # (0, lo) due to transpose
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update the accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multipleof(start_kv, BLOCK_SIZE_KV)

        # compute qk
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if stage == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block = QK_block - m_ij[:, None]
        else:
            # update the running max
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            # compute softmax_star on QK_block
            QK_block = QK_block * softma_scale - m_ij[:, None]

        # compute the P_block
        P_block = tl.math.exp(QK_block)

        # get the current sum
        l_ij = tl.sum(P_block, 1)

        # correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)

        # apply the correction factor to the previous l_i
        l_i = l_i * alpha + l_ij

        # load the V_blovk
        V_block = tl.load(V_block_ptr)

        P_block = P_block.to(tl.float16)

        # O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block) # optimised O_block += P_block @ V_block

        # update running max
        m_i = m_ij

        # advance K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))  # V[SEQ_LEN, HEAD_DIM]
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))  # K[HED_DIM, SEQ_LEN
        
    return O_block, l_i, m_i


@triton.jit
def _attn_fwd(
        Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        M, 
        softmax_scale, 
        O, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        stride_Q_batch,
        stride_Q_head,
        stride_Q_seq,
        stride_Q_dim,
        stride_K_batch,
        stride_K_head,
        stride_K_seq,
        stride_K_dim,
        stride_V_batch,
        stride_V_head,
        stride_V_seq,
        stride_V_dim
        stride_O_batch,
        stride_O_head,
        stride_O_seq,
        stride_O_dim,
        BATCH_SIZE,
        NUM_HEADS: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        STAGE: tl.constexpr,
):
    # which block in the seq_length to process
    block_index_q = tl.program_id(0)

    index_batch_head = tl.program_id(1)

    # which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS

    # position of head within the batch 
    index_head = index_batch_head % NUM_HEADS

    # get the (SEQ_LEN, HEAD_DIM) block in the Q, K, V by indexing with the index_head and index_batch
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch,
        index_head.to(tl.int64) * stride_Q_head 
    )

    # get base ptr to the query block this program deals with
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # as each program deals with only a particular block of query offset by block_index_q * BLOCK_SIZE_Q
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )

    # get base ptr to the value block this program will start with
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0) # each program deals with all the blocks of value thus offset is 0
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0)
    )

    # get base ptr to the key block this program will start with
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_K_seq, stride_K_dim), # to access the transpose of Key blocks, interchange the strides
        offsets=(0, 0) # each program deals with all the blocks of key thus offset is 0
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV), # change block shape according to transpose
        order=(0, 1)
    )

    # output block will be filled in same manner as query block
    O_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # as each program deals with only a particular block of query
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )

    # coming to the blocks this program will execute
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # m_i - the running maximum - we have one for each query in query block, initalize with -inf
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    # l_i - the running sum - we have one for each query in query block, later used to normalize softmax
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0 # add 1 for numeric stability

    # accumulator for the output which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # load the query block (it will stay in the SRAM thorughout)
    Q_block = tl.load(Q_block_ptr)

    # stage = 3 is causal else 1

    if stage == 1 or stage == 3:
        # this step runs for non-causal or for blocks to the left of diagonal in causal attention
        Q_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q
            softmax_scale
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4-stage,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    
    if stage == 3:
        # this step runs for the blocks to the right of the diagonal in the causal attention
        Q_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    # logexpsum trick - store m_i as m_i + log(l_i) - used in backward pass
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))



class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q = Q.shape[-1]
        HEAD_DIM_K = K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        # intializa output with zero
        O = torch.zeros_like(Q)
        stage = 3 is causal else 1

        # defin the grid
        grid = lambda args: (
            torch.cdiv(SEQ_LEN, args['BLOCK_SIZE']),  # which group of queries we are going to work with 
            BATCH_SIZE * NUM_HEADS, # which head of which batch element we are going to work with
            1 # Z dim
        )

        # num of parallel programs = (BATCH_SIZE * NUM_HEADS * HEAD_DIM_Q)

        # logsumexp for the backward pass: one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        __attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            M=M,
            softmax_scale=softmax_scale,
            O=O,
            stride_Q_batch=Q.stride[0],
            stride_Q_head=Q.stride[1],
            stride_Q_seq=Q.stride[2],
            stride_Q_dim=Q.stride[3],
            stride_K_batch=K.stride[0],
            stride_K_head=K.stride[1],
            stride_K_seq=K.stride[2],
            stride_K_dim=K.stride[3],
            stride_V_batch=V.stride[0],
            stride_V_head=V.stride[1],
            stride_V_seq=V.stride[2],
            stride_V_dim=V.stride[3],
            stride_O_batch=O.stride[0],
            stride_O_head=O.stride[1],
            stride_O_seq=O.stride[2],
            stride_O_dim=O.stride[3],
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM_Q=HEAD_DIM_K,
            STAGE=stage,
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O



def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    K = (torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    V = (torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())

    softmax_scale = 1 / (HEAD_DIM ** 0.5) # normalizing factor
    dO = torch.rand_like(Q) # used in backward pass

    # naive attention implementation
    MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device='cuda'))
    P = (Q @ K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK==0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_o = P @ V
    ref_o.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None

    # compare
    atol=1e-2
    rtol=0
    assert torch.isclose(ref_o, tri_out, atol=atol, rtol=rtol)
    assert torch.isclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.isclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.isclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)
