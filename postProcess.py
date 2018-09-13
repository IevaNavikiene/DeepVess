import os

def postProcess(stack_path):
    import matlab.engine
    import os
    eng = matlab.engine.start_matlab()
    eng.cd('api/resources/preprocessing/DeepVess', nargout=0)

    # TODO change all those strange file names into something consistent
    path, h5file_path = os.path.split(os.path.abspath(stack_path))
    output_path = h5file_path.split("noMotion-Ch4-8bit-")[1].split('V_fwd.mat')[0]
    main_filename_part = h5file_path[:-3]
    mask_file_path = main_filename_part, "-masked.tif"

    full_mat_filename = os.path.join(path, main_filename_part, "-V_fwd.mat")
    eng.postProcess(path, h5file_path + ".h5", output_path + ".mat", nargout=0)
    eng.postProcess(stack_path, output_path, mask_file_path, full_mat_filename)