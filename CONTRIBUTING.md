# Contribution Guidelines

1. Focus on performance: Any new layer must have a quantized equivalent.
2. Maintain zero-dependency status for the `runtime` module where possible.
3. Add unit tests for every new feature.
4. Use PEP8 coding standards.

## Development Setup
```bash
git clone https://github.com/user/edgelogic
cd edgelogic
pip install -e ".[dev]"
pytest
```
