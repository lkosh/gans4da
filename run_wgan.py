import os
import shutil
import subprocess
train_path = '../data/GTSRB/Final_Training/Images/'
dirs = ['tmp_data', 'tmp_data/Images', 'wgan_models']
for d in dirs:
    if not os.path.exists(d):
        os.mkdir(d)
        
# os.mkdir('tmp_data')
# os.mkdir('tmp_data/Images')
# os.mkdir('wgan_models')
for i in range( 43):
    print ('#'*100 + '\n' + str(i) + '\n' + '#'*100)
        

    src = train_path+'0'*(5-len(str(i))) + str(i)
    dst = './tmp_data/Images/' + '0'*(5-len(str(i))) + str(i)
    #if not os.path.exists(dst):

    shutil.copytree(src, dst)
    if not os.path.exists('wgan_models/class_'+str(i)):
        os.mkdir('wgan_models/class_'+str(i))
        
    subprocess.check_call(['python','./original_main.py', '--dataset' ,'folder', '--dataroot', 'tmp_data', '--cuda', '--experiment', 'wgan_models/class_'+str(i)])
    shutil.rmtree(dst)
    
    