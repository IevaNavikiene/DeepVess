def postProcess(stack_path):
    import matlab.engine
    import os
    eng = matlab.engine.start_matlab()
    eng.cd('api/resources/preprocessing/DeepVess', nargout=0)

    # TODO change all those strange file names into something consistent
    path, h5file_path = os.path.split(os.path.abspath(stack_path))
    output_path = h5file_path.split("noMotion-Ch4-8bit-")[1].split('V_fwd.mat')[0]
    eng.postProcess(path, h5file_path + ".h5", output_path + ".mat", nargout=0)