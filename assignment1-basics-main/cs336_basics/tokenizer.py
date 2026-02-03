import regex as re
from collections.abc import Iterable

class BPRTokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):

        # 建立双向映射，方便查表
        self.vocab = vocab
        self.id_to_byte = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}

        # 将合并规则转换为Rank字典
        # BPE 编码时，必须优先应用在训练阶段较早出现的合并规则
        self.merges = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []

        # 构建特殊 Token 的正则表达式
        if self.special_token:

            sorted_special = sorted(self.special tokens, key=len, reverse=True)

            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        # GPT-2 官方预分词正则表达式
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        if not self.special_regex:
            return self._encode_text_segment(text)
        
        # 处理含有特殊标记的复杂文本
        tokens = []

        # last_pos 用于记录上一次匹配结束的位置，帮助定位"特殊标记"之间的"缝隙"
        last_pos = 0

        # 使用 finditer 遍历文本中所有符合特殊标记模式的匹配项
        # finditer 提供了 match.start() 和 match.end()
        for match in self.special_regex.finditer(text):

            # 提取并处理"前置普通文本"
            pre_text = text[last_pos:match.start()]

            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            
            # 处理"当前特殊标记"
            special_tok = match.group()
            tokens.append(self.byte_to_id[special_tok.encode("utf-8")])

            last_pos = match.end()

        # 处理"收尾文本"
        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))

        return tokens

        
    def _encode_text_segment(self, text: str) -> list[int]:
        ids = []
        pre_tokens = self.gpt2_pat.findall(text)

        for p_tok in pre_tokens:
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]

            while len(byte_parts) >= 2:
                best_pair = None
                min_rank = float('inf')

                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i], byte_parts[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                
                if best_pair is None:
                    break

                # 执行合并操作
                new_byte_parts = []
                i = 0
                while i < len(byte_parts):
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                byte_parts = new_byte_parts
            
            # 将合并到极限后的所有字节块转换为词表中的 ID
            for part in byte_parts:
                ids.append(self.byte_to_id[part])

        return ids

    def decode(self, ids: list[int]) -> str:
        # 根据 ID 查表找回字节块
        byte_segments = [self.id_to_byte[i] for i in ids]
        # 将所有字节快按顺序拼接成一个完整的字节流
        full_bytes = b"".join(byte_segments)
        # 将字节流解码为 UTF-8 字符串
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)