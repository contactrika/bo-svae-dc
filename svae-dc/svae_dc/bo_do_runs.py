#
# A quick utility script to launch multiple BO runs locally.
#
import os
import sys


def main():
    if len(sys.argv) < 5:
        print('Specify: env name, kernel type, run id strt, num runs')
        assert(False)
    device_str = sys.argv[1]
    gpus = None
    if device_str != 'cpu': gpus = device_str.split('_')
    print('Will use GPUs', gpus)
    envnm = sys.argv[2]
    if envnm == 'YumiVel':
        envnm = 'YumiVel-v2'
        ctrl = 'WaypointsVelPolicy'
    elif envnm == 'FrankaVel':
        envnm = 'FrankaVel-v2'
        ctrl = 'WaypointsVelPolicy'
    elif envnm == 'FrankaTorque':
        envnm = 'FrankaTorque-v2'
        ctrl = 'WaypointsMinJerkPolicy'
    elif envnm == 'Daisy':
        envnm = 'DaisyCustomLong-v0'
        #ctrl = 'DaisyGait11DPolicy'
        ctrl = 'DaisyGait27DPolicy'
    else:
        print('Unsupported env', envnm); assert(False)
    knl = sys.argv[3]
    rnstrt = int(sys.argv[4])
    nruns = int(sys.argv[5])
    nnchkpt = os.path.expanduser(sys.argv[6]) if len(sys.argv) >= 7 else None
    tr = '--bo_num_init=2 --bo_num_trials=40'
    if knl == 'Random': tr = '--bo_num_init=42 --bo_num_trials=0'
    gpu_idx = 0.0
    for rn in range(rnstrt,rnstrt+nruns):
        cmd = 'python bo_main.py '+tr
        if gpus is not None: cmd += ' --gpu='+gpus[int(gpu_idx)]
        cmd += ' --run_id={:d} --bo_kernel_type={:s} '.format(rn, knl)
        cmd += ' --env_name={:s} --controller_class={:s}'.format(envnm, ctrl)
        if nnchkpt is not None:
            #cmd += ' --svae_dc_override_good_th={:0.4f} '.format(0.35)
            cmd += ' --svae_dc_checkpt={:s} &'.format(nnchkpt)
        else:
            cmd += ' &'  # run in background
        print('cmd', cmd)
        os.system(cmd)
        gpu_idx += 0.5  # 2 jobs per GPU
        if int(gpu_idx>=len(gpus)): gpu_idx = 0.0


if __name__ == '__main__':
    main()
