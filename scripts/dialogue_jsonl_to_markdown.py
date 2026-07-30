"""対話JSONLを、発話内容を変更せず閲覧用Markdownへ変換する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SPEAKER_LABELS = {
    "A": "A（システム）",
    "B": "B（ユーザー）",
}


def convert_file(source: Path, destination: Path) -> None:
    rows: list[str] = []
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{source}:{line_number}: [話者, 発話] ではありません")
        speaker, utterance = value
        if speaker not in SPEAKER_LABELS or not isinstance(utterance, str):
            raise ValueError(f"{source}:{line_number}: 話者または発話が不正です")
        # JSONLと同様に1ターン1行。末尾の空白2個はMarkdownの改行指定。
        rows.append(f"**{SPEAKER_LABELS[speaker]}:** {utterance}  ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n")


def default_output_for_file(source: Path) -> Path:
    """dialogues配下なら、同じデータセットのmarkdown配下へ対応づける。"""
    source = source.resolve()
    for parent in source.parents:
        if parent.name == "dialogues":
            return parent.parent / "markdown" / source.relative_to(parent).with_suffix(".md")
    return source.parent / "markdown" / source.with_suffix(".md").name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="対話JSONLファイル、またはdialoguesディレクトリ")
    parser.add_argument(
        "--out", help="出力先。省略時は同じデータセットのmarkdownディレクトリ"
    )
    args = parser.parse_args()

    source = Path(args.input)
    if source.is_file():
        output = Path(args.out) if args.out else default_output_for_file(source)
        convert_file(source, output)
        print(f"converted=1 output={output}")
        return
    if not source.is_dir():
        raise FileNotFoundError(source)

    output = Path(args.out) if args.out else source.parent / "markdown"
    files = sorted(source.rglob("*.jsonl"))
    for path in files:
        destination = output / path.relative_to(source).with_suffix(".md")
        convert_file(path, destination)
    print(f"converted={len(files)} output={output}")


if __name__ == "__main__":
    main()
