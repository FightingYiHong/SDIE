from DRL.TD3 import TD3
from utils.drl_utils import create_directory, scale_action
from utils. new_env import *
from utils.options import args_parser
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

args = args_parser()
def input(figure,GT):#输入原图和GT的路径
    figure = cv2.imread(figure, cv2.IMREAD_GRAYSCALE)
    GT = cv2.imread(GT, cv2.IMREAD_GRAYSCALE)
    train=1
    env = fed(figure,train)
    dir = args.ckpt_dir_TD3 + args.dataset + '/'
    agent = TD3(alpha=0.0003, beta=0.0003, state_dim=env.observation_space,
                action_dim=9, actor_fc1_dim=400, actor_fc2_dim=300,
                critic_fc1_dim=50, critic_fc2_dim=300, ckpt_dir=dir, gamma=0.99,
                tau=0.005, action_noise=0.05, policy_noise=0.1, policy_noise_clip=0.5,
                delay_time=2, max_size=8192, batch_size=32)

    create_directory(path=dir, sub_path_list=['Actor', 'Critic1', 'Critic2', 'Target_actor',
                                              'Target_critic1', 'Target_critic2'])
    agent.load_models(1000)
    done = False
    observation=figure
    count=0
    while not done:
        count+=1
        action = agent.choose_action(np.array(observation), train=train)
        observation_, reward, done = env.step(np.array(action),observation,GT)
        observation = observation_
        if count >= 4:
            break
    iou=SAM(observation, GT)
    cv2.imwrite('Enhancement.jpg',observation)#保存Enhancement的图片
    return iou#评价指标
if __name__ == '__main__':
    input('FSSD-12/Steel_Am/Images/Steel_Am_1.jpg','FSSD-12/Steel_Am/GT/Steel_Am_1.png')#从前端接收的原图片以及GT的路径