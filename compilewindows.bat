
@echo USAGE:
@echo ------
@echo Open the "Developer Command Prompt for VS<version>"
@echo For 32-bit Python, run: "C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\Tools\vsvars32.bat"
@echo For 64-bit Python, run: "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall" amd64
@echo Then finally run compilewindows.bat (this file)
@echo.

@pause

cl gco_source\LinkedBlockList.cpp /link /out:LinkedBlockList.obj
cl gco_source\graph.cpp /link /out:graph.obj
cl gco_source\maxflow.cpp /link /out:maxflow.obj
cl gco_source\GCoptimization.cpp /link /out:GCoptimization.obj
cl cgco.cpp /link /out:cgco.obj
cl LinkedBlockList.obj graph.obj maxflow.obj GCoptimization.obj cgco.obj /link /dll /out:libcgco.dll

cl /EHsc test_wrapper.cpp libcgco.lib
