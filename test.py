import numpy as np
import matplotlib.pyplot as plt

class MarkovProcess:
    #初始化，输入参数
    def __init__(self, states, transition_matrix, reward_matrix_pool, reward_matrix_others,alpha,gamma):
        self.states = states  
        self.transition_matrix = transition_matrix 
        self.reward_matrix_pool = reward_matrix_pool
        self.reward_matrix_others = reward_matrix_others
        self.current_state = 0
        self.alpha=alpha
        self.gamma=gamma

    #新挖出一个区块并结算收益
    def step(self):
       if(self.current_state in states and self.current_state != 3):
        current_index = self.states.index(self.current_state)
        next_state = np.random.choice(self.states, p=self.transition_matrix[current_index])
        reward_pool = self.get_reward_pool(current_index, self.states.index(next_state))
        reward_others = self.get_reward_others(current_index, self.states.index(next_state))
        #如果出现分叉相同长度的情况，特殊考虑收益
        if(self.current_state == -1):
            action = np.random.choice([0,1,2],p=[alpha,gamma*(1-alpha),(1-gamma)*(1-alpha)])
            reward_pool = 2-action
            reward_others = action
        self.current_state = next_state
        return next_state, reward_pool, reward_others
       else:
        action=np.random.choice([-1,1],p=[1-alpha,alpha])
        next_state = self.current_state+action
        reward_pool = 1 if (action == -1) else 0
        reward_others = 0
        self.current_state = next_state
        return next_state, reward_pool, reward_others
        
    #从pool收益矩阵中读取对应收益
    def get_reward_pool(self, current_index, next_index):
        
        return self.reward_matrix_pool[current_index][next_index]
    
    #从others收益矩阵中读取对应收益
    def get_reward_others(self, current_index, next_index):
        
        return self.reward_matrix_others[current_index][next_index]

    #进行一轮模拟，给定区块总数，返回自私矿池和其他成员总收益和历史记录
    def simulate(self, steps):
        
        history = [(self.current_state, 0, 0)]  
        total_reward_pool = 0
        total_reward_others = 0

        for _ in range(steps):
            next_state, reward_pool, reward_others = self.step()
            history.append((next_state, reward_pool, reward_others))
            total_reward_pool += reward_pool
            total_reward_others += reward_others
        
        return history, total_reward_pool, total_reward_others



if __name__ == "__main__":
 plt.figure(figsize=(10, 6))
 x=np.linspace(0,0.5,50)
 y=x
 plt.plot(x,y,label='Honest mining')
 alpha_values=[0.1,0.2,0.3,0.4,0.45,0.47]
 for gamma in [0,0.5,1]:
  R_pool_sim_results=[]

  for alpha in [0.1,0.2,0.3,0.4,0.45,0.47]:
    states = [-1,0,1,2,3]
    #states表示私有链领先区块数，-1表示分叉后公有链私有链等长，领先数大于等于3时在马尔可夫过程中特殊处理
    transition_matrix = [
        [0, 1, 0, 0, 0],
        [0, 1-alpha, alpha, 0, 0],  
        [1-alpha, 0, 0, alpha, 0],  
        [0, 1-alpha, 0, 0, alpha]  
    ]
    #reward在新挖出区块时结算，若reward待定，则当前步骤结算为0
    reward_matrix_pool = [
        [0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0],  
        [0, 2, 0, 0, 0]   
    ]

    reward_matrix_others = [
        [0, 0, 0, 0, 0],  
        [0, 1, 0, 0, 0],  
        [0, 0, 0, 0, 0],  
        [0, 0, 0 ,0, 0]   
    ]

    #transition仅考虑转入领先3区块概率，不考虑转出，分叉转化为唯一主链有多种可能的转换情况，计算收益时拆分考虑，即reward_matrix第一行可以为任意值
    markov_process = MarkovProcess(states, transition_matrix, reward_matrix_pool, reward_matrix_others,alpha,gamma)
    result, total_reward_pool, total_reward_others = markov_process.simulate(100000)  
    R_pool_sim= total_reward_pool/(total_reward_pool+total_reward_others)
    R_pool_sim_results.append(R_pool_sim)
  marker = 'o' if gamma == 0 else ('+' if gamma == 0.5 else '*')
  plt.scatter(alpha_values, R_pool_sim_results, marker=marker, label=f'γ = {gamma}(sim)',s=70)


 alpha=np.linspace(0,0.5,50)
 gamma_values = [0, 0.5, 1]  

 line_styles = {
    0: 'solid',     
    0.5: 'dashed',  
    1: 'dotted'     
}
 for gamma in gamma_values:
    R_pool = (alpha * (1 - alpha)**2 * (4 * alpha + gamma * (1 - 2 * alpha)) - alpha**3) / (1 - alpha * (1 + (2 - alpha) * alpha))
    plt.plot(alpha, R_pool, label=f'γ = {gamma}', linestyle=line_styles[gamma])
 plt.xlabel('Alpha')
 plt.ylabel('Relative pool revenue')
 plt.grid(linewidth=0.5)
 plt.xticks(np.arange(0, 0.5, 0.05))
 plt.yticks(np.arange(0, 1, 0.1))
 plt.legend()
 plt.show()