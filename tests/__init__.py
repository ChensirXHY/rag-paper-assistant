"""测试包。

把 ``tests/`` 标记为包（而不是靠 pytest 的 rootdir 自动插入 ``sys.path``），
有两个实际好处：

1. 测试模块名带包前缀（``tests.test_loader``），与 ``src.loader`` 之类的
   业务模块名**不会互相遮蔽** —— 这是 pytest 里很常见的踩坑点；
2. 在 ``tests/conftest.py`` 里把项目根目录写进 ``sys.path`` 后，
   ``import config`` / ``import src.xxx`` 在任意工作目录下都能成功，
   不必要求"必须从项目根目录运行 pytest"。
"""
