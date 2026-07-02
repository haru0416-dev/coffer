"""Shared asciicast-v2 composer for the demo record scripts (no terminal recorder, no
hand-editing): the record scripts run the REAL commands, capture their REAL output, and
use this to script only the typing cadence around it. Render with agg:

    agg --theme dracula --font-size 16 demo/<name>.cast demo/<name>.gif
"""
import json

TYPE_MS = 0.014  # per typed character
PROMPT = "\x1b[1;32m$\x1b[0m "
GREY, CYAN = "90", "36"


class Cast:
    def __init__(self, cols=132, rows=42, title=""):
        self.cols, self.rows, self.title = cols, rows, title
        self.events = []
        self.now = 0.0

    def emit(self, text, dt=0.0):
        self.now += dt
        self.events.append([round(self.now, 4), "o", text])

    def type_line(self, line, lead=0.15, tail=0.25, color=None):
        """Simulate typing one shell line (comment or command) at the prompt."""
        self.emit(PROMPT, lead)
        shown = f"\x1b[{color}m{line}\x1b[0m" if color else line
        # type per character over len*TYPE_MS, in a few chunks so the cast stays small
        chunk = max(1, len(shown) // 24)
        for i in range(0, len(shown), chunk):
            self.emit(shown[i : i + chunk], TYPE_MS * chunk)
        self.emit("\r\n", tail)

    def comment(self, line, tail=1.4):
        self.type_line(f"# {line}", tail=tail, color=GREY)

    def command(self, line, tail=0.25):
        self.type_line(line, tail=tail, color=CYAN)

    def output(self, text, pause_after=2.0):
        self.emit(text.replace("\n", "\r\n") + "\r\n", 0.12)
        self.now += pause_after

    def clear(self, dt=0.3):
        self.emit("\x1b[2J\x1b[H", dt)

    def write(self, path):
        header = {
            "version": 2,
            "width": self.cols,
            "height": self.rows,
            "title": self.title,
            "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
        }
        with open(path, "w") as f:
            f.write(json.dumps(header) + "\n")
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return self.now, len(self.events)
