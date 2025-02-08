import torch
import json
import random
from Y_NETS.resonet_v2 import Y_NET
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

data_paths = {"train_cha":r"E:\PythonProjects\ResoNet\dataset\arc-agi_training_challenges.json",
              "train_sol":r"E:\PythonProjects\ResoNet\dataset\arc-agi_training_solutions.json",
              "test_cha":r"E:\PythonProjects\ResoNet\dataset\arc-agi_test_challenges.json",
              "eval_sol":r"E:\PythonProjects\ResoNet\dataset\arc-agi_evaluation_solutions.json",
              "eval_cha":r"E:\PythonProjects\ResoNet\dataset\arc-agi_evaluation_challenges.json"}

with open(data_paths['eval_cha'],'r') as eval_cha:
    eval_cha = json.load(eval_cha)
with open(data_paths['eval_sol'],'r') as eval_sol:
    eval_sol = json.load(eval_sol)

with open(data_paths['train_cha'],'r') as train_chr:
    train_cha = json.load(train_chr)
with open(data_paths['train_sol'],'r') as train_sol:
    train_sol = json.load(train_sol)


print(torch.cuda.is_available())
problem = random.sample(train_cha.keys(),k=20)

"""
y_net = Y_NET(grid_shape=(1,30,30),channels=1,batch_size=32,device='cuda')

dataset = y_net.data_loader(data_pairs=train_cha,loader='train',batch_size=32)
loss_en,loss_de = y_net.train(dataset,batch_size=32,epochs=1000,en_lr=0.001,de_lr=0.001)

x_axe = torch.linspace(1,len(loss_en),len(loss_en))
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].plot(x_axe,loss_en)
axs[1].plot(x_axe,loss_de)

plt.show()

torch.save(y_net.encoder.state_dict(),"save_y_nets//encoder_v2.pth")
torch.save(y_net.decoder.state_dict(),"save_y_nets//decoder_v2.pth")
"""
problem = ['5582e5ca', 'd8c310e9', 'f35d900a', '09629e4f', '82819916', 'eb5a1d5d', 'b527c5c6', 'f25fbde4', '44d8ac46', 'f8b3ba0a', '445eab21', 'ce602527', 'c59eb873', '3de23699', '73251a56', '5117e062', '8d510a79', 'a78176bb', 'b7249182', 'c909285e']

arc = train_cha[problem[11]]
sol =  train_sol[problem[11]]

sol = torch.tensor(sol[0])
model = Y_NET(grid_shape=(1,30,30),channels=1,batch_size=1,device='cuda')



pred, loss_en,loss_de = model.test_eval(arc_problem=arc,epochs=300,activation=150,en_lr=0.005,de_lr=0.005)
x_axe = torch.linspace(1,len(loss_en),len(loss_en))

output = []
for i in range(pred.shape[0]):
    try:
        output.append(model.grid_scaler(grid=pred[i], mode='down'))
    except Exception as e:
        print(e)

output = [arc['test'][0]['input']] + output + [sol]


# fig, axs = plt.subplots(nrows=1, ncols=2)
#
# axs[0].plot(x_axe,loss_en)
# axs[1].plot(x_axe,loss_de)

# plt.show()


model.cplot(grid=output,sequence=True,pairs=0,save_path="test_1")

#model = Y_NET(grid_shape=(1, 30, 30), channels=1, batch_size=1, device='cuda')
#model.encoder.load_state_dict(torch.load('save_y_nets\\encoder_v2.pth'))
#model.decoder.load_state_dict(torch.load('save_y_nets\\decoder_v2.pth'))

def group():
    for i in problem:
        try:
            save_path = f"outputs\\ite_2\\test_{i}.jpg"

            arc = train_cha[i]
            sol = train_sol[i]

            sol = torch.tensor(sol[0])
            model = Y_NET(grid_shape=(1, 30, 30), channels=1, batch_size=1, device='cuda')


            output, loss_en, loss_de = model.test_fit(arc_problem=arc, epochs=300, activation=200, outputs=10,
                                                      en_lr=0.005,
                                                      de_lr=0.005)
            x_axe = torch.linspace(1, len(loss_en), len(loss_en))

            # fig, axs = plt.subplots(nrows=1, ncols=2)
            #
            # axs[0].plot(x_axe,loss_en)
            # axs[1].plot(x_axe,loss_de)

            # plt.show()

            outputs = []
            for i in output:
                outputs.append(i[1])
            outputs = [output[0][0]] + outputs + [sol]


            model.cplot(grid=outputs, sequence=True, pairs=0, save_path=save_path,show=False)

        except Exception as e:
            print(e)


