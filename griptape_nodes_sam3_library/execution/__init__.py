"""Execution-only modules, declared in the manifest's `execution_modules`.

Files here may import the library's execution dependencies at module scope. The
orchestrator never imports them; a worker imports them eagerly at library load, so a
missing dependency fails once, at startup, with the real traceback. Node code reaches
them through `self.execution_module("<file stem>")`.
"""
