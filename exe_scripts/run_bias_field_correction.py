import os, glob
from tqdm.auto import tqdm
from modules.bias_field_correction import run_n4_bias_correction

if __name__ == '__main__':

    log_file = '/Users/yschoi/data/FLAIR/err_bias_field_correction.txt'
    path_src = '/zdisk/users/ext_user_03/01_yschoi/project_01_FVH_detection/01_data/03_flair_preproc_v02_toServer'
    path_dst = '/zdisk/users/ext_user_03/01_yschoi/project_01_FVH_detection/01_data/03_flair_preproc_v02_toServer'

    list_subject = sorted([_dir for _dir in os.listdir(path_src) if not _dir.startswith('.')])

    for _subject in tqdm(list_subject):
        for _fn in [_fn for _fn in os.listdir(os.path.join(path_src, _subject)) if not(_fn.startswith('.')) and ('brain.nii' in _fn)]:
            input_dir = os.path.join(path_src,_subject, _fn)
            output_dir = os.path.join(path_dst, _subject)
            mask_dir = glob.glob(os.path.join(path_src, _subject, '*brainMask.nii.gz'))[0]
            out_name = _subject
            
            print('='*20)
            try:
                run_n4_bias_correction(in_path=input_dir, out_dir=output_dir, out_name=out_name, external_mask_path=mask_dir)
            except Exception as e:
                # 에러 로그 파일에 누적 기록
                with open(log_file, "a") as f:
                    f.write("="*60 + "\n")
                    f.write(f"Input : {input_dir}\n")
                    f.write(f"Output: {output_dir}\n")
                    f.write(f"Error : {str(e)}\n\n")
                print(f"[ERROR] 변환 실패 → 로그 저장: {input_dir}")
            print('\n')
    
    ## run example
    ## (flair_preproc) yschoi@Youngseokui-MacBookPro 09_FHV % python -m 01_exe_files.02_run_bias_field_correction
