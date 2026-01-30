# 2026 AI Specialist 1일, 2일차

## 강의자료 코드 Overview
- 코드가 존재하지 않음

## Chapter 2 Dataset (코드 중심)
```python
* tokenizer 받기
tokenizer = tiktoken.get_encoding("gpt2")

* encoding
integers = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
integers = tokenizer.encode(tet, allowed_special={"<|endoftext|>"})
--> 이 경우, integers는 list 따라서 tensor로 변환이 필요
integers_tensor = torch.tensor(integers)
torch.tensor()는 encoding된 vector를 tensor로 바꾸는 역할을 한다. 

* tokenizer.decode(tensor)는 tensor를 입력으로 받아서 string으로 변환한다.
type(integers) -> list
type(integer_tensor) -> tensor

--> token을 decoding한다. 
decoding_text = tokenizer.decode(integer_tensor)
tokenizer.decode는 출력이 string이다. 

* list vs set vs tuple
[1, 2, 3] vs {1, 2, 3} vs (1, 2, 3)

## for 문으로 돌리기
for i in range(0, len(token_ids) - max_length, stride):
    input_chunk = token_ids[i: i+max_length]

    # 데이터를 Tensor로 변환해서 리스트에 저장
    self.input_ids.append(torch.tensor(input_chunk))
```


## Chapter 3 Exercise Attention (코드 중심)
* Causual Attention에서 Query, Key, Values를 받기 위한 코드. 
```python
## 쿼리, 키, 벨로를 위한 선형 투영 레이어
self.query  = nn.Linear(d_in, d_out, bias)
self.key    = nn.Linear(d_in, d_out, bias)
self.values = nn.Linear(d_in, d_out, bias)

## 과적합 방지를 위한 드롭아웃
self.dropout = nn.Dropout(dropout)

## Causal Mask 생성 버퍼
self.register_buffer(
    'mask',
    torch.griu(torch.ones(context_length, context_lenght), diagonal=1)
)

## forward 함수에서 query, key, value 구하기
def forward(self, x):
    batch, num_tocken, d_in = x.shape

    query = self.query(x)
    key = self.key(x)
    values = self.values(x)

    # attention score를 계산한다. QK_Transpose
    attention_score = query @ key.transpose(1,2)
    
    # mask가 1인 위치를 -무한대로 채운다. 
    attention_score.maked_fill_(
        self.mask.bool()[:num_tokens, :num_tokens],
        -torch.inf
    )

    attention_weights = torch.softmax(
        attention_score / key.shape[1]**0.5,
         dimx=-1
    )

    # 가중치중 무작위로 0으로 만든다. 
    attention_weights = self.dropouot(attention_weights)

    # 가중치를 기반으로 value를 계산한다.
    context_vector = attention_weights @ values

    return context_vector

## MultiheadAttention을 CausalAttention을 이요해서 구하는 방법
class MultiheadAttention_with_CausalAttention(nn.Module):
    def __init__(self, din, dout, context_lenght, drop_out, num_heads, bias=False):
        super.__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(din, dout, context_length, dropout, bias) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x)] for head in self.heads], dim=-1)

torch.manual_seed(123)

context_length = batch.shape[1]

d_in, d_out = 3, 2
mha = MultiheadAttention(din, dout, context_length, drop_out=0.0, num_heads = 2)

# 독립적으로 multiheadAttention을 구한다. 
class MultiheadAttention(nn.Module):
    def __init__(self, din, dout, context_lenght, dropout, num_heads, bias=False):
        super.__init__()
        assert (dout % num_head == 0), "dout must be dividable with num_head"

        self.dout=dout
        self.num_heads = num_heads
        self.head_dim = dout // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
    def forward(self, x):
        b, num_tokens, din = x.shape

        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        ## multihead
        
```


