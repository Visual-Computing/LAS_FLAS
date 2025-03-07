from typing import List, Dict, Any, Sequence, Callable, Tuple


def main():
    table = Table(('A', 'Header with more words', 'c'))
    table.line(A=3.141, c='hello')
    table.line(A=202.71828182, Header_with_more_words=1234567)
    table.line(A=2.1, c='blabla')
    table.line(A='long string', Header_with_more_words=32)
    table.line(A=12345, Header_with_more_words='something')
    print(table)


class Table:
    def __init__(
            self, headers: Sequence[Any], float_precision: int = 3, column_space: int = 2,
            lower_keys: bool = True
    ):
        self._lower_keys = lower_keys
        self._string_formatter = StringFormatter(float_precision=float_precision)
        self._headers: Sequence[Tuple[str, int]] = [self._string_formatter(h, 0, 0) for h in headers]
        self._normed_headers: List[str] = [self._norm_header(h) for h in headers]
        self._lines: List[Dict[str, Any]] = []
        self._column_space = column_space

    def line(self, **kwargs):
        for key in kwargs:
            if key not in self._normed_headers:
                raise KeyError('Add value for header "{}", but this header does not exist. '
                               'Valid headers are: {}'.format(key, ', '.join(self._normed_headers)))
        self._lines.append(kwargs)

    def __repr__(self):
        column_width = {nh: h[1] for nh, h in zip(self._normed_headers, self._headers)}
        sep = ' ' * self._column_space
        longest_float = {nh: 0 for nh in self._normed_headers}
        longest_int = {nh: 0 for nh in self._normed_headers}
        for line in self._lines:
            for header, value in line.items():
                str_len = self._string_formatter(value, 0, 0)[1]
                column_width[header] = max(column_width[header], str_len)
                if Table.has_value_type(value, float):
                    longest_float[header] = max(longest_float[header], str_len)
                if Table.has_value_type(value, int):
                    longest_int[header] = max(longest_int[header], str_len)

        header_line = sep.join(
            Table.ljust(h, l, column_width[nh]) for nh, (h, l) in zip(self._normed_headers, self._headers)
        )
        lines = [header_line]

        for line in self._lines:
            str_line = []
            for header in self._normed_headers:
                col_width = column_width[header]
                if header in line:
                    value = line[header]
                    str_value, str_len = self._string_formatter(value, longest_float[header], longest_int[header])
                    str_line.append(Table.ljust(str_value, str_len, col_width))
                else:
                    str_line.append(' ' * col_width)
            lines.append(sep.join(str_line))
        return '\n'.join(lines)

    def _norm_header(self, header):
        header, _header_len = self._string_formatter(header, 0, 0)
        header = header.replace(' ', '_')
        if self._lower_keys:
            header = header.lower()
        return header

    @staticmethod
    def ljust(value: str, current_length: int, length: int):
        return value + ' ' * (length - current_length)

    @staticmethod
    def has_value_type(value, t):
        return isinstance(value, t)


class StringFormatter:
    def __init__(self, float_precision: int = 3):
        self._float_fmt_string = '{{:<.{:.0f}f}}'.format(float_precision)
        self.formatters: Dict[Any, Callable[[Any, int, int], Tuple[str, int]]] = {
            float: self._format_float,
            int: StringFormatter._format_int
        }
        self._np = None
        try:
            import numpy as np
            self._np = np
        except ImportError:
            pass

    def __call__(self, value, longest_float_hint: int, longest_int_hint: int):
        value = self._normalize_numpy_types(value)
        formatter = self.formatters.get(type(value))
        if formatter is not None:
            return formatter(value, longest_float_hint, longest_int_hint)
        str_value = str(value)
        return str_value, len(str_value)

    def _normalize_numpy_types(self, value):
        if self._np is not None:
            if self._np.issubdtype(type(value), self._np.floating):
                return float(value)
            if self._np.issubdtype(type(value), self._np.integer):
                return int(value)
        return value

    def _format_float(self, value: float, longest_float_hint: int, _longest_int_hint: int):
        str_value = self._float_fmt_string.format(value).rjust(longest_float_hint)
        return str_value, len(str_value)

    @staticmethod
    def _format_int(value: float, _longest_float_hint: int, longest_int_hint: int):
        str_value = str(value).rjust(longest_int_hint)
        return str_value, len(str_value)


if __name__ == '__main__':
    main()
