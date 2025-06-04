How to install:

```bash
pip install mrc_parser-0.0.1-py3-none-any.whl
```

How to import

```python
import mrc_parser

# OR

from mrc_parser import MRC_Parser, BoxnetTF
```

Troubleshooting:
- Make sure the model file is specified and is a .tflite file
- For some reason, pip doesn't like when the .whl file name is changed, so this should be exactly `mrc_parser-0.0.1-py3-none-any.whl`