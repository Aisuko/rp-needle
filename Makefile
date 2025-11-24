install:
	uv pip install -e .

docs:
	pdoc needlehaystack -o docs

docs-server:
	pdoc needlehaystack -p 8080