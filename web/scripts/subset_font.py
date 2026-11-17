"""Subset an OpenType math font to the given characters and print it as base64 WOFF2.

usage: python3 subset_font.py FONT.otf CHARS.txt

Keeps the MATH table (and the glyph variants/assemblies it references) so that
radicals, fractions and stretchy delimiters still render. The family is renamed
to "LabMath" because the result is a modified version of the original font.
"""
import base64
import io
import sys

from fontTools import subset
from fontTools.ttLib import TTFont


def main(font_path: str, chars_path: str) -> None:
    text = open(chars_path, encoding="utf-8").read()
    options = subset.Options()
    options.layout_features = ["*"]
    options.flavor = "woff2"
    options.name_IDs = ["*"]
    options.notdef_outline = True
    options.drop_tables = [t for t in options.drop_tables if t != "MATH"]
    font = TTFont(font_path, recalcTimestamp=False)  # deterministic output
    sub = subset.Subsetter(options)
    sub.populate(unicodes={ord(c) for c in text})
    sub.subset(font)
    for rec in font["name"].names:
        if rec.nameID in (1, 3, 4, 6, 16, 17):
            new = {1: "LabMath", 3: "LabMath-Subset", 4: "LabMath", 6: "LabMath-Regular",
                   16: "LabMath", 17: "Regular"}[rec.nameID]
            rec.string = new
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    sys.stdout.write(base64.b64encode(buf.getvalue()).decode("ascii"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
