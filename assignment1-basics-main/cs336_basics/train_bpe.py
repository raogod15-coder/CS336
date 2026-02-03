def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # 初始化基础词表
    vocab = {i: bytes([i]) for i in range(256)}

    # 计算需要进行的合并次数
    # 目标词表大小 = 基础字节数 (256) + 特殊 Token 数 + 需要新生成的 Token 数
    num_merges = vocab_size - 256 - len(special_tokens)

    # 读取语料，并按特殊 Token 分割
    with open(input_path,"r",encoding="utf-8") as f:
        text = f.read()

    # special_tokens不参与语料统计
    if special_tokens:

        special_regex = "|".join(re.escape(t) for t in special_tokens)

        parts = re.split(f"({special_regex})", text)

        train_segment = [p for p in parts if p not in special_tokens]
    else:
        train_segment = [text]

    # 预分词并统计词频
    # GPT-2 不允许跨越类型合并，保护空格
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # raw_counts: 存储每个"单词"(预分词后的结果)及其出现频率
    raw_counts = Counter()
    for segment in train_segment:
        words = gpt2_pat.findall(segment)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode("utf-8"))] += 1

    # 构建高效数据结构以支持快速合并
    # words_list: 存储每个单词的字节列表  使用 list 而不是 tuple 因为BPE合并会修改单词内部结构
    # counts_list: 存储对应单词的频率
    words_list = []
    counts_list = []
    for word_tuple, freq in raw_counts.items():
        words_list.append(list(word_tuple))
        counts_list.append(freq)

    # stats: 存储所有可能的相邻字节对(pair) 及其全局出现的频率
    # 结构: {(byte_a, byte_b): frequency}
    stats = defaultdict(int)

    # indices: 倒排索引 存储 pair -> {包含该 pair 的单词在 words_list 中的下标集合}
    indices = defaultdict(set)

    # 初始化'stats'和'indices'
    # 遍历所有唯一的单词
    for idx,word in enumerate(words_list):
        freq = counts_list[idx]

        for i in range(len(word) - 1):
            pair = (word[i],word[i+1])
            stats[pair] += freq
            indices[pair].add(idx)
            
    merges = []

    # 迭代合并
    for _ in range(num_merges):
        if not stats:
            break
        
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        if stats[best_pair] <= 0:
            break
        
        # 记录
        merges.append(best_pair)
        # 创建新的 Token
        new_token = best_pair[0] + best_pair[1]
        
        # 获取需要更新的单词
        relevant_indices = list(indices[best_pair])

        # 遍历并更新所有受影响的单词
        for idx in relevant_indices:
            word = words_list[idx]
            freq = counts_list[idx]

            # 扫描当前单词，找到所有'best_pair'的出现位置
            i=0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    if i > 0:
                        prev_pair = (word[i-1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:   
                            del stats[prev_pair]
                        
                    if i < len(word) - 2:
                        next_pair = (word[i+1], word[i+2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]
                    # 修改单词结构：将(word[i], word[i+1]) 替换为 new_token
                    word[i] = new_token
                    del word[i+1]

                    if i > 0:
                        new_prev = (word[i-1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    if i < len(word) - 1:                            
                        new_next = (word[i],word[i+1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                    
                else:
                    i += 1
            
        # 移除已完全合并的'best_pair'
        if best_pair in stats: del stats[best_pair]            
        if best_pair in indices: del indices[best_pair]

    # 构建最终的词表
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]

    # 添加特殊 Token
    for s_tok in special_tokens:
        s_byte = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_byte
        
    return vocab, merges