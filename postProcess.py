import os

def postProcess(h5_path):
    import matlab.engine
    import os
    eng = matlab.engine.start_matlab()
    eng.cd('api/resources/preprocessing/DeepVess', nargout=0)

    # TODO change all those strange file names into something consistent
    path, h5file_path = os.path.split(os.path.abspath(h5_path))
    main_filename_part = h5file_path[:-3]
    output_path = os.path.join(path, main_filename_part.split("noMotion-Ch4-8bit-")[1] + '.mat')
    mask_file_path = os.path.join(path, main_filename_part+ "-masked.tif")

    full_mat_filename = os.path.join(path, main_filename_part+ "-V_fwd.mat")

    print(h5_path, output_path, mask_file_path, full_mat_filename,'stack_path, output_path, mask_file_path, full_mat_filename')
    #eng.postProcess(path, h5file_path + ".h5", output_path + ".mat", nargout=0)
    eng.postProcess(os.path.join(path, h5_path), output_path, mask_file_path, full_mat_filename, nargout=0)
