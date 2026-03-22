from pathlib import Path
import json
import random

PROJECT_ROOT = Path("/content/cs-exam-coach-qlora")
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "你是一位資工考試助理。"
    "回答必須使用繁體中文，並固定輸出四個段落："
    "1.一句話定義 2.核心觀念 3.常見誤區 4.簡單例子。"
    "內容要清楚、考試導向、避免太口語。"
)

samples = [
    {
        "user": "什麼是 semaphore？",
        "assistant": """1. 一句話定義
Semaphore 是用來控制多個 process 或 thread 存取共享資源的同步工具。

2. 核心觀念
它本質上是一個計數器，會搭配 wait(P) 與 signal(V) 操作來管理資源數量。若資源可用，wait 會讓計數器減 1；若資源不足，執行緒可能被阻塞。signal 則會釋放資源，讓計數器加 1。

3. 常見誤區
很多人把 semaphore 和 mutex 當成完全一樣，但 mutex 通常強調互斥；semaphore 可以允許多個執行緒同時存取有限數量的資源。

4. 簡單例子
假設印表機池有 3 台，可以設一個初值為 3 的 counting semaphore，最多允許 3 個工作同時使用印表機。"""
    },
    {
        "user": "什麼是 deadlock？",
        "assistant": """1. 一句話定義
Deadlock 是多個程序彼此等待對方釋放資源，結果全部都無法繼續執行的狀態。

2. 核心觀念
死結通常與資源競爭有關，經典條件包含 mutual exclusion、hold and wait、no preemption、circular wait。四個條件同時成立時，就可能發生死結。

3. 常見誤區
很多人把 deadlock 和 starvation 混在一起。deadlock 是大家都卡死；starvation 則是某些程序長期拿不到資源，但系統其他部分可能仍在運作。

4. 簡單例子
程序 A 拿到資源 X 等待 Y，程序 B 拿到資源 Y 等待 X，兩邊互等，就形成 deadlock。"""
    },
    {
        "user": "什麼是 paging？",
        "assistant": """1. 一句話定義
Paging 是把虛擬記憶體與實體記憶體切成固定大小區塊來管理的記憶體配置方式。

2. 核心觀念
虛擬位址空間被切成 pages，實體記憶體被切成 frames，page table 用來記錄 page 對應到哪個 frame。它能避免 external fragmentation，但可能有 internal fragmentation。

3. 常見誤區
很多人以為 paging 完全沒有碎裂，其實它只是避免外部碎裂，仍可能有內部碎裂。

4. 簡單例子
若 page 大小是 4KB，一個 10KB 的程式會被分成 3 個 pages，最後一個 page 可能沒有完全用滿。"""
    },
    {
        "user": "什麼是 process 與 thread 的差別？",
        "assistant": """1. 一句話定義
Process 是資源配置的基本單位，thread 是 CPU 排程的基本執行單位。

2. 核心觀念
同一個 process 底下的 threads 共享程式碼區、資料區與開啟的資源，但各自有自己的暫存器與 stack。建立與切換 thread 的成本通常低於 process。

3. 常見誤區
很多人只記得 thread 比較輕量，卻忘了共享記憶體也代表同步問題會更多。

4. 簡單例子
瀏覽器可以是一個 process，而裡面處理 UI、網路請求、背景任務的不同工作可由多個 threads 分工。"""
    },
    {
        "user": "什麼是 binary search？",
        "assistant": """1. 一句話定義
Binary search 是在已排序資料中，每次把搜尋範圍砍半的查找演算法。

2. 核心觀念
每一步比較中間值與目標值，若目標較小就往左半搜尋，較大就往右半搜尋，因此時間複雜度為 O(log n)。

3. 常見誤區
很多人忘記 binary search 的前提是資料必須先排序；若資料未排序，直接使用會得到錯誤結果。

4. 簡單例子
在 1, 3, 5, 7, 9, 11 中找 7，先看中間值 5，因為 7 比 5 大，所以往右半邊找，很快就能找到 7。"""
    },
    {
        "user": "什麼是 stack 與 queue 的差別？",
        "assistant": """1. 一句話定義
Stack 是後進先出，queue 是先進先出。

2. 核心觀念
Stack 常用 push 與 pop，queue 常用 enqueue 與 dequeue。Stack 適合處理遞迴、函式呼叫；queue 適合處理排隊、排程與 BFS。

3. 常見誤區
很多人只背 LIFO 與 FIFO，卻不會連結到實際應用場景，考題常會從應用反問資料結構。

4. 簡單例子
洗盤子常像 stack，後放上去的盤子先拿下來；排隊買票則像 queue，先到的人先服務。"""
    },
]

random.seed(42)
random.shuffle(samples)

split_idx = max(1, int(len(samples) * 0.8))
train_samples = samples[:split_idx]
eval_samples = samples[split_idx:]

def to_record(item: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ]
    }

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

write_jsonl(DATA_DIR / "train.jsonl", [to_record(x) for x in train_samples])
write_jsonl(DATA_DIR / "eval.jsonl", [to_record(x) for x in eval_samples])

print("done")
print(f"train: {len(train_samples)}")
print(f"eval : {len(eval_samples)}")
print(f"saved to: {DATA_DIR}")
