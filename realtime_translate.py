#!/usr/bin/env python3
"""
WhisperLiveKit 实时识别 + Index-Translate 免费API 翻译
用法：
  1. 先启动 WhisperLiveKit（必须带 --pcm-input，否则服务端期待编码音频而非裸 PCM）：
       wlk --model large-v3-turbo --lan ja --pcm-input
  2. 文件模式（音频需为 16kHz 单声道 PCM WAV）：
       python realtime_translate.py --file speech.wav
     非 16kHz/单声道的文件先用 ffmpeg 转换：
       ffmpeg -i in.mp3 -ar 16000 -ac 1 -acodec pcm_s16le out.wav
  3. 实时模式（录系统声音的 monitor 旁路，不影响外放，Ctrl+C 结束）：
       python realtime_translate.py --monitor            # 默认输出设备
       python realtime_translate.py --monitor <SINK名>   # 指定输出设备
       python realtime_translate.py --monitor --gain 4   # 音量小丢识别时加增益
     注意：录的是整个系统的声音，看目标内容时其它应用请静音。
     屏幕底部实时显示 dBFS 音量（说话时约 -30 ~ -15 为宜），据此调 --gain。
  4. Chrome 单标签模式（只采一个标签的声音，其它标签/应用不受影响）：
       python realtime_translate.py --tab                # 建虚拟声卡 wlk-tab 并采它的 monitor
     配套步骤：
       a) 用 Chrome 扩展 Tab Audio Selector：在目标标签点扩展 ->
          Output Device 选 “WLK Tab”（首次需 Grant Permission 授权一次）
       b) 脚本会自动创建 wlk-tab（设备名 WLK Tab）和外放回路，
          声音照常能听到；停止时自动卸载，系统完全还原
       c) 每次会话结束后需在扩展里重新选一次 WLK Tab（授权不会失效）

说明：
  - WLK 的 /asr 协议是"客户端推音频、服务端返结果"：本脚本把音频切成 0.5s
    的 PCM 块推给服务端，收到空二进制帧 b"" 表示音频结束。
  - lines[] 末尾是"随语音增长的活动行"（start 不变、text 变长），转正需停顿超过
    服务端 --pause-segmentation-seconds。本脚本跟踪其增长：遇句末标点即切成完整句
    打印并整句翻译；间隙对新增部分做增量翻译（灰色两行预览：当前句原文+累积译文
    实时滚动，停顿时切回音量表）；停止增长 2 秒或换行时把尾巴整段终翻收尾。
"""

import argparse
import asyncio
import json
import math
import shutil
import signal
import subprocess
import sys
import time
import unicodedata
import wave
from collections import deque
from datetime import datetime

import numpy as np
import requests
import websockets

# ========== 配置 ==========
WLK_WS_URL = "ws://localhost:8000/asr"          # WhisperLiveKit WebSocket
TARGET_LANG = "zh"                              # 翻译目标语言（ISO 639-1）
INDEX_API = "https://index-translate.bilibili.com/v1/chat/completions"
MODEL = "Index-Translate-35B-A3B"
SAMPLE_RATE = 16000                             # WLK 固定输入采样率
CHUNK_SECONDS = 0.5                             # 每块音频时长（16000 字节）
USE_PROXY = False                               # True=翻译请求走 http_proxy/https_proxy 环境变量
MIN_TRANSLATE_INTERVAL = 0.3                    # 两次翻译请求的最小间隔（秒），防限流
MT_DRAIN_TIMEOUT = 60                           # 结束时等待翻译队列清空的最长时间（秒）
TAIL_DEBOUNCE = 0.8                             # 活动句增量翻译的最小间隔（秒）
COMMIT_IDLE_S = 2.0                             # 文本停止增长多少秒视为该行定稿
MAX_LINE_LEN = 160                              # 无标点长串的兜底切句长度（字符）
SENT_END_CHARS = set("。！？!?")                # 句末标点
LINE_CLOSERS = set("」』）》”’\"')")             # 句末标点后的收尾引号
LANG_NAMES = {                                  # prompt 里用的语言名（比 ISO 码更自然）
    "zh": "中文", "en": "English", "ja": "日本語", "yue": "粤语",
    "ko": "한국어", "fr": "Français", "de": "Deutsch",
    "es": "Español", "ru": "Русский",
}
# --- Chrome 单标签模式（--tab）：专用虚拟声卡 ---
TAB_SINK_NAME = "wlk-tab"                       # pactl sink 名（稳定，脚本内部用）
TAB_SINK_DESC = "WLK Tab"                       # 设备描述（Chrome 扩展里选这个名字）
TAB_LOOPBACK_SRC = f"{TAB_SINK_NAME}.monitor"   # loopback 的源，也用于定位模块
TAB_LOOPBACK_TAG = "wlk-tab-lb"                 # 传给 module-loopback 的标记参数
# ==========================

_session = requests.Session()
_session.trust_env = USE_PROXY   # False = 忽略 http_proxy/https_proxy（B 站接口直连）
# 接口对 python-requests 的 UA 返回 412（反爬），伪装成浏览器
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
    ),
})


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def translate(text: str, target: str = "zh") -> str:
    """调用 Index-Translate 免费 API（失败重试一次）"""
    if not text.strip():
        return ""

    target_name = LANG_NAMES.get(target, target)
    prompt = f"请将以下文本翻译为{target_name}，直接输出翻译结果，不要进行任何解释。\n\n{text}"

    last_error = None
    for attempt in range(2):
        try:
            resp = _session.post(
                INDEX_API,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(1.0)
    return f"[翻译失败: {last_error}]"


def read_wav_chunks(path: str):
    """校验并按 0.5s 块产出 PCM s16le 数据"""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise SystemExit(
                f"[错误] {path} 不是 16-bit PCM WAV。转换：\n"
                f"  ffmpeg -i {path} -ar 16000 -ac 1 -acodec pcm_s16le out.wav"
            )
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
            raise SystemExit(
                f"[错误] 需要 {SAMPLE_RATE} Hz 单声道 WAV"
                f"（当前 {wf.getframerate()} Hz / {wf.getnchannels()} 声道）。转换：\n"
                f"  ffmpeg -i {path} -ar 16000 -ac 1 -acodec pcm_s16le out.wav"
            )
        block = int(SAMPLE_RATE * CHUNK_SECONDS)
        while True:
            data = wf.readframes(block)
            if not data:
                return
            yield data


async def send_file(ws, path: str):
    """把 WAV 切块推给服务端，结束时发空帧 b""（协议约定的 EOF）"""
    n_chunks = 0
    for chunk in read_wav_chunks(path):
        await ws.send(chunk)
        n_chunks += 1
        await asyncio.sleep(0)   # 让出事件循环，避免堵住接收任务
    await ws.send(b"")
    print(f"[{ts()}] 音频发送完毕（{n_chunks} 块），等待服务端完成 ...")


def resolve_monitor_sink(sink: str | None) -> str:
    """解析要录制的 monitor 源：不指定则取默认输出设备的 monitor"""
    if not sink:
        proc = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, timeout=5,
        )
        default_sink = proc.stdout.strip()
        if proc.returncode != 0 or not default_sink:
            raise SystemExit(
                f"[错误] 无法获取默认输出设备: {proc.stderr.strip() or 'pactl 无输出'}"
            )
        return f"{default_sink}.monitor"
    # 用户可能传 sink 名（alsa_output...）或 monitor 源名（...monitor）
    return sink if sink.endswith(".monitor") else f"{sink}.monitor"


# ---------- Chrome 单标签模式（--tab）：虚拟声卡管理 ----------

def _pactl(*args: str, check: bool = True) -> str:
    """执行 pactl，返回 stdout；check=True 时失败即退出（错误信息带 stderr）"""
    proc = subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=10)
    if check and proc.returncode != 0:
        raise SystemExit(
            f"[错误] pactl {' '.join(args)} 失败: "
            f"{proc.stderr.strip() or proc.stdout.strip() or '未知错误'}"
        )
    return proc.stdout


def pactl_list(kind: str) -> list[list[str]]:
    """解析 `pactl list short <kind>`（kind: sinks/sources/modules），按 tab 分列"""
    out = _pactl("list", "short", kind)
    return [line.split("\t") for line in out.splitlines() if "\t" in line]


def tab_sink_exists() -> bool:
    return any(cols[1] == TAB_SINK_NAME for cols in pactl_list("sinks"))


def find_module_index(needle: str) -> str | None:
    """按模块参数找模块编号（pactl 会原样回显加载时的参数）"""
    for cols in pactl_list("modules"):
        if len(cols) >= 3 and needle in cols[2]:
            return cols[0]
    return None


def ensure_tab_sink() -> None:
    """创建 wlk-tab 虚拟声卡（已存在则复用）；rate=48000 保外放音质，重采样交给 parec"""
    if tab_sink_exists():
        return
    _pactl(
        "load-module", "module-null-sink",
        f"sink_name={TAB_SINK_NAME}",
        f"sink_properties=device.description={TAB_SINK_DESC}",
        "rate=48000", "channels=2",
    )
    print(f"[{ts()}] 已创建虚拟声卡: {TAB_SINK_NAME}（Chrome 里显示为 “{TAB_SINK_DESC}”）")


def ensure_tab_loopback() -> None:
    """把 wlk-tab.monitor 复制回默认输出（字幕运行时照样能听到）；已存在则复用"""
    if find_module_index(TAB_LOOPBACK_TAG) is not None:
        return
    proc = subprocess.run(
        ["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5
    )
    default_sink = proc.stdout.strip()
    if proc.returncode != 0 or not default_sink:
        raise SystemExit("[错误] 无法获取默认输出设备，不能建立外放回路")
    _pactl(
        "load-module", "module-loopback",
        f"source={TAB_LOOPBACK_SRC}", f"sink={default_sink}",
        f"tag={TAB_LOOPBACK_TAG}", "latency_msec=50",
    )
    print(f"[{ts()}] 已建立外放回路: {TAB_LOOPBACK_SRC} -> {default_sink}")


def teardown_tab_modules() -> None:
    """卸载 loopback 与虚拟声卡（完全还原；其上的流会被 PipeWire 移回默认 sink）"""
    for needle, what in (
        (TAB_LOOPBACK_TAG, "外放回路"),
        (f"sink_name={TAB_SINK_NAME}", "虚拟声卡"),
    ):
        idx = find_module_index(needle)
        if idx is not None:
            _pactl("unload-module", idx, check=False)
            print(f"[{ts()}] 已卸载{what}（模块 {idx}）")


class MonitorCapture:
    """parec 子进程：录 PipeWire sink 的 monitor（旁路 tap，不影响外放）"""

    def __init__(self, sink: str):
        self.sink = sink
        self.proc: asyncio.subprocess.Process | None = None
        self.errors: deque = deque(maxlen=20)

    async def start(self):
        # 用 pulse 层的 parec 录 monitor（pw-record 无法链接 monitor 节点）；
        # PipeWire 自行重采样，直接产出 16kHz 单声道 s16 裸流（--raw）
        self.proc = await asyncio.create_subprocess_exec(
            "parec",
            "-d", self.sink,
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--raw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self):
        while line := await self.proc.stderr.readline():
            self.errors.append(line.decode(errors="replace").strip())

    def stop(self):
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()


async def monitor_sender(ws, capture: MonitorCapture, gain: float, preview: dict):
    """从 monitor 读块持续推流（可选增益）；采集结束时发 b"" 收尾"""
    chunk_bytes = int(SAMPLE_RATE * CHUNK_SECONDS * 2)   # s16 = 2 字节/采样
    amplify = gain != 1.0
    try:
        while True:
            data = await capture.proc.stdout.read(chunk_bytes)
            if not data:          # 采集进程退出 → 音频流结束
                break
            samples = np.frombuffer(data, dtype=np.int16)
            if samples.size:
                if amplify:
                    # 提升音量帮 VAD 过阈值；float 中间量 + 限幅防削波失真
                    boosted = np.clip(
                        samples.astype(np.float32) * gain, -32768, 32767
                    ).astype(np.int16)
                    samples = boosted
                    data = boosted.tobytes()
                # 音量表显示增益后的电平 = 服务端实际收到的，据此调 --gain
                rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                preview["db"] = (
                    20.0 * math.log10(rms / 32768.0) if rms > 0 else -120.0
                )
            await ws.send(data)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        returncode = await capture.proc.wait()
        if returncode not in (0, -15) and capture.errors:   # -15 = 我们发的 SIGTERM
            tail = "; ".join(list(capture.errors)[-3:])
            print(f"[{ts()}] 采集进程异常退出(code={returncode}): {tail}")
        try:
            await ws.send(b"")   # EOF：让服务端收尾并回 ready_to_stop
        except Exception:
            pass


class LineState:
    """一行转写的客户端侧状态：跟踪活动行增长、按句切分、增量翻译"""
    __slots__ = (
        "speaker", "full_text", "committed", "delta_offset", "trans",
        "gen", "last_growth", "last_mt", "finalized", "delta_pending",
    )

    def __init__(self, speaker, text: str, now: float):
        self.speaker = speaker if speaker is not None else "?"
        self.full_text = text       # 服务端该行最新文本（不 strip，保偏移稳定）
        self.committed = 0          # 已按完整句打印/送翻的字符偏移
        self.delta_offset = 0       # 已送增量翻译的字符偏移
        self.trans = ""             # 未切出尾巴的累积译文（仅灰色预览）
        self.gen = 0                # 代数：切句/定稿后使在途增量结果失效
        self.last_growth = now
        self.last_mt = 0.0
        self.finalized = False
        self.delta_pending = False  # 同一行只允许一个增量翻译在途（防重复段）


def take_sentence(text: str, start: int):
    """从 start 切出一个完整句（句末标点+收尾引号）；无则按长度兜底。
    返回 (chunk, new_offset)；无可切时返回 (None, start)"""
    i = start
    while i < len(text):
        j = -1
        for pos in range(i, len(text)):
            if text[pos] in SENT_END_CHARS:
                j = pos + 1
                while j < len(text) and text[j] in LINE_CLOSERS:
                    j += 1
                break
        if j < 0:
            break
        if j - i >= 2:              # 忽略 "！" 等单字符碎片
            return text[i:j], j
        i = j
    # 长度兜底：超过 MAX_LINE_LEN 仍无句号 → 在后半段的顿号/逗号/空格处切
    if len(text) - start > MAX_LINE_LEN:
        cut = -1
        for sep in ("、", "，", ",", " "):
            p = text.rfind(sep, start + MAX_LINE_LEN // 2)
            if p > cut:
                cut = p
        if cut > start:
            return text[start:cut + 1], cut + 1
    return None, start


def cut_sentences(st: LineState, mt_queue: asyncio.Queue):
    """活动行里已出现的完整句 → 逐句送整句终翻（成对打印）"""
    while not st.finalized:
        chunk, off = take_sentence(st.full_text, st.committed)
        if chunk is None:
            break
        st.committed = off
        st.delta_offset = off
        st.trans = ""
        st.gen += 1
        piece = chunk.strip()
        if piece:
            mt_queue.put_nowait(("chunk", st.speaker, piece))


def finalize_line(st: LineState, mt_queue: asyncio.Queue):
    """行定稿（停顿超时/新行出现/会话收尾）：先按句切，余下尾巴整段终翻"""
    if st.finalized:
        return
    cut_sentences(st, mt_queue)
    st.finalized = True
    st.gen += 1
    remainder = st.full_text[st.committed:].strip()
    st.committed = len(st.full_text)
    st.delta_offset = st.committed
    st.trans = ""
    if remainder:
        mt_queue.put_nowait(("chunk", st.speaker, remainder))


def flush_all(states: dict, mt_queue: asyncio.Queue):
    for st in states.values():
        finalize_line(st, mt_queue)


async def receive_and_translate(
    ws, mt_queue: asyncio.Queue, states: dict, preview: dict
) -> bool:
    """消费服务端 JSON：跟踪活动行增长，按句切分 + 增量翻译入队"""
    async for message in ws:
        try:
            data = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            continue

        msg_type = data.get("type")
        if msg_type == "config":
            continue
        if msg_type == "ready_to_stop":
            clear_preview(preview)
            print(f"[{ts()}] 服务端处理完成 (ready_to_stop)")
            flush_all(states, mt_queue)
            return True
        if data.get("error") or data.get("status") == "error":
            clear_preview(preview)
            print(f"[{ts()}] 服务端错误: {data.get('error') or data.get('status')}")
            flush_all(states, mt_queue)
            return False

        now = time.monotonic()
        active_key = None
        for line in data.get("lines") or []:
            raw = line.get("text") or ""
            if not raw.strip():
                continue           # 静音段（speaker=-2, text=''）
            key = (line.get("speaker"), line.get("start"))
            active_key = key       # lines[] 末尾即随语音增长的活动行
            st = states.get(key)
            if st is None:
                # 新行首次出现：不立即打印（修复"首碎片即定稿、后半永不翻"的缺陷）
                states[key] = LineState(line.get("speaker"), raw, now)
            elif raw != st.full_text:
                if st.finalized:
                    if len(raw) <= len(st.full_text):
                        continue   # 转正后的行文本冻结，重复下发忽略
                    st.finalized = False   # 定稿后又增长（停顿判早了）→ 重开跟踪
                if len(raw) < st.delta_offset:
                    # 对齐回滚文本变短：放弃在途增量，从已切点重新累积
                    st.delta_offset = st.committed
                    st.trans = ""
                    st.gen += 1
                st.full_text = raw
                st.last_growth = now

        preview["active_key"] = active_key

        # 非活动行不再增长（新行出现/服务端已转正）→ 尾巴立即送终翻
        for k2, st2 in states.items():
            if not st2.finalized and k2 != active_key:
                finalize_line(st2, mt_queue)

        if active_key is not None:
            st = states[active_key]
            if not st.finalized:
                cut_sentences(st, mt_queue)
                tail = st.full_text[st.delta_offset:]
                if (
                    tail
                    and not st.delta_pending
                    and now - st.last_mt >= TAIL_DEBOUNCE
                ):
                    st.last_mt = now
                    st.delta_pending = True
                    mt_queue.put_nowait(
                        ("delta", active_key, st.gen, tail, len(st.full_text))
                    )

        draw_preview(preview, states)

    clear_preview(preview)
    print(f"[{ts()}] 连接被服务端关闭")
    flush_all(states, mt_queue)
    return False


async def idle_finalizer(states: dict, preview: dict, mt_queue: asyncio.Queue):
    """文本停止增长超 COMMIT_IDLE_S 的行 → 定稿
    （停顿期间服务端不再推消息，靠此定时器收尾）"""
    while True:
        await asyncio.sleep(0.5)
        now = time.monotonic()
        changed = False
        for st in list(states.values()):
            if not st.finalized and now - st.last_growth >= COMMIT_IDLE_S:
                finalize_line(st, mt_queue)
                changed = True
        if changed:
            draw_preview(preview, states)


def clear_preview(preview: dict):
    """清掉已绘制的预览行（可能多行），非 TTY 静默"""
    if not sys.stdout.isatty():
        return
    rows = preview.get("rows", 0)
    if rows:
        sys.stdout.write(f"\033[{rows}A\033[J")
        preview["rows"] = 0
        sys.stdout.flush()


def _disp_width(s: str) -> int:
    """字符串显示列数：宽字符（中日韩全角）按 2 列，其余按 1 列"""
    return sum(
        2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        for c in s
    )


def _fit(s: str, max_cols: int) -> str:
    """按显示列数从尾部保留内容（预览随语音增长，尾部最新），超出时前加 …"""
    if _disp_width(s) <= max_cols:
        return s
    out = []
    w = 0
    for c in reversed(s):
        cw = 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        if w + cw > max_cols - 1:   # 预留省略号的 1 列
            break
        out.append(c)
        w += cw
    return "…" + "".join(reversed(out))


def draw_preview(preview: dict, states: dict):
    """行内灰色预览：活动句最新原文 + 累积译文（两行）；空闲时音量表"""
    if not sys.stdout.isatty():
        return
    rows = preview.get("rows", 0)
    if rows:
        sys.stdout.write(f"\033[{rows}A")   # 上移到预览首行
    sys.stdout.write("\033[J")              # 清到屏尾重画
    # 预览行必须保证不换行：宽字符 1 字 = 2 列，超宽折行会让"上移行数"错位，
    # 旧的折行残留在屏幕上清不掉 → 越积越多刷屏。按终端列数动态截断根治。
    cols = shutil.get_terminal_size((100, 24)).columns
    budget = max(cols - 4, 16)              # 扣掉 "| "/"~ " 前缀与安全余量
    n = 0
    st = states.get(preview.get("active_key"))
    if st is not None and not st.finalized:
        tail = _fit(st.full_text[st.committed:].strip(), budget)
        if tail:
            sys.stdout.write(f"\x1b[90m| {tail}\x1b[0m\n")
            n += 1
        trans = _fit(st.trans.strip(), budget)
        if trans:
            sys.stdout.write(f"\x1b[90m~ {trans}\x1b[0m\n")
            n += 1
    if n == 0 and preview.get("db") is not None:
        vol = (
            f"[音量] {preview['db']:6.1f} dBFS "
            f"(gain={preview.get('gain', 1.0):g})"
        )
        sys.stdout.write(f"\x1b[90m{_fit(vol, budget)}\x1b[0m\n")
        n += 1
    preview["rows"] = n
    sys.stdout.flush()


async def mt_worker(mt_queue: asyncio.Queue, states: dict, preview: dict):
    """串行翻译：chunk=完整句终翻（成对打印），delta=活动句增量（进预览）"""
    last_call = 0.0
    while True:
        item = await mt_queue.get()
        if item is None:      # 结束哨兵
            return
        wait = MIN_TRANSLATE_INTERVAL - (time.monotonic() - last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        if item[0] == "chunk":
            _, speaker, text = item
            translated = await asyncio.to_thread(translate, text, TARGET_LANG)
            last_call = time.monotonic()
            clear_preview(preview)
            print(f"[{ts()}] {speaker} 原文: {text}")
            print(f"[{ts()}] {speaker} 译文: {translated}")
        else:                 # delta：活动句增量翻译，追加到灰色预览
            _, key, gen, text, end = item
            translated = await asyncio.to_thread(translate, text, TARGET_LANG)
            last_call = time.monotonic()
            st = states.get(key)
            if st is not None:
                st.delta_pending = False    # 结果无论是否过期都要解除在途标记
                if not st.finalized and st.gen == gen:
                    st.trans += translated
                    st.delta_offset = end
                    draw_preview(preview, states)


async def listen_and_translate(
    file_path: str | None, monitor_sink: str | None, gain: float = 1.0,
    tab: bool = False,
):
    print(f"[{ts()}] 正在连接 WhisperLiveKit ({WLK_WS_URL}) ...")

    # proxy=None：websockets>=14 会读 http_proxy/https_proxy 环境变量，
    # 不显式禁用的话本机 ws 请求也会被代理劫走（返回代理机的 404）
    async with websockets.connect(WLK_WS_URL, proxy=None) as ws:
        # 首条消息必须是 config
        first = json.loads(await ws.recv())
        if first.get("type") != "config":
            print(f"[{ts()}] 意外的首条消息: {first}")
            return

        if not first.get("useAudioWorklet"):
            print(
                f"[{ts()}] [错误] 服务器未启用 --pcm-input（useAudioWorklet=false），"
                "期待的是编码音频而非裸 PCM。\n"
                "请重启服务器：wlk --model large-v3-turbo --lan ja --pcm-input"
            )
            return

        backend = first.get("context", {}).get("backend", "?")
        capture = None
        preview: dict = {"rows": 0, "active_key": None, "db": None, "gain": gain}
        if file_path:
            print(f"[{ts()}] 已连接（backend={backend}），开始发送音频 ...\n")
            send_task = asyncio.create_task(send_file(ws, file_path))
        else:
            if tab:
                # Chrome 单标签模式：建虚拟声卡 + 外放回路，采它的 monitor
                ensure_tab_sink()
                ensure_tab_loopback()
                sink = TAB_LOOPBACK_SRC
            else:
                sink = resolve_monitor_sink(monitor_sink)
            capture = MonitorCapture(sink)
            await capture.start()
            if tab:
                print(
                    f"[{ts()}] 已连接（backend={backend}），模式: Chrome 单标签采集\n"
                    f"  1) 在目标标签用 Tab Audio Selector 把 Output Device 选为 “{TAB_SINK_DESC}”\n"
                    f"  2) 只有被选中的标签进字幕；其它标签/应用不受影响\n"
                    f"  3) 声音经回路照常外放；停止时自动还原虚拟设备\n"
                )
            else:
                print(
                    f"[{ts()}] 已连接（backend={backend}），"
                    f"正在采集系统声音（{sink}, gain={gain:g}），Ctrl+C 停止 ...\n"
                )
            send_task = asyncio.create_task(
                monitor_sender(ws, capture, gain, preview)
            )

        states: dict = {}
        mt_queue: asyncio.Queue = asyncio.Queue()
        mt_task = asyncio.create_task(mt_worker(mt_queue, states, preview))
        idle_task = asyncio.create_task(idle_finalizer(states, preview, mt_queue))
        recv_task = asyncio.create_task(
            receive_and_translate(ws, mt_queue, states, preview)
        )

        try:
            # shield：Ctrl+C 取消的是本任务，recv_task 继续存活，
            # 便于 finally 里等服务端走完 ready_to_stop 再关连接
            await asyncio.shield(recv_task)
        finally:
            if capture is not None:
                # 停采集 → sender 读到 EOF 会自动发 b"" 让服务端收尾
                capture.stop()
                try:
                    await asyncio.wait_for(send_task, timeout=10)
                except (TimeoutError, asyncio.TimeoutError):
                    send_task.cancel()
            elif not send_task.done():
                send_task.cancel()
            if tab:
                # 尽早释放音频设备：后面翻译排空最多还要 60s，期间若被强杀也不会漏还原
                teardown_tab_modules()
            if not recv_task.done():
                try:
                    await asyncio.wait_for(recv_task, timeout=5)
                except (TimeoutError, asyncio.TimeoutError):
                    recv_task.cancel()
                except Exception:
                    pass
            idle_task.cancel()
            flush_all(states, mt_queue)   # 兜底：异常路径漏掉的尾巴也送终翻
            # 等翻译队列清空（每句最多 ~21s），再结束
            await mt_queue.put(None)
            try:
                await asyncio.wait_for(mt_task, timeout=MT_DRAIN_TIMEOUT)
            except (TimeoutError, asyncio.TimeoutError):
                mt_task.cancel()
                print(f"[{ts()}] 翻译队列超时未清空，已放弃剩余 {mt_queue.qsize()} 条")


def main():
    parser = argparse.ArgumentParser(
        description="WhisperLiveKit 转写 + Index-Translate 翻译",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--file",
        help="16kHz 单声道 PCM WAV 文件（其它格式先用 ffmpeg 转换）",
    )
    mode.add_argument(
        "--monitor", nargs="?", const="", metavar="SINK",
        help="实时录系统声音（默认输出设备的 monitor；可选指定 sink 名，Ctrl+C 结束）",
    )
    mode.add_argument(
        "--tab", action="store_true",
        help="只采 Chrome 单标签：需先用 Tab Audio Selector 把该标签输出指到 “WLK Tab”",
    )
    parser.add_argument(
        "--gain", type=float, default=1.0, metavar="N",
        help="monitor 模式采集增益（如 2.0/4.0；配合屏幕上的 dBFS 音量显示调整）",
    )
    args = parser.parse_args()
    if args.gain <= 0:
        parser.error("--gain 必须大于 0")

    # SIGTERM/SIGHUP 默认会直接掐死进程、跳过 finally（虚拟声卡就不还原了）。
    # 转成 KeyboardInterrupt → 走与 Ctrl+C 完全相同的优雅收尾路径（含 teardown）。
    def _graceful_stop(signum, frame):
        raise KeyboardInterrupt

    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _graceful_stop)
        except (ValueError, OSError):
            pass

    try:
        asyncio.run(listen_and_translate(args.file, args.monitor, args.gain, tab=args.tab))
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
