import torch
import json
import random
from Y_NETS.resonet_v2 import Y_NET
import matplotlib.pyplot as plt

def load_data(data_paths):
    with open(data_paths['eval_cha'], 'r') as eval_cha:
        eval_cha = json.load(eval_cha)
    with open(data_paths['eval_sol'], 'r') as eval_sol:
        eval_sol = json.load(eval_sol)
    with open(data_paths['train_cha'], 'r') as train_chr:
        train_cha = json.load(train_chr)
    with open(data_paths['train_sol'], 'r') as train_sol:
        train_sol = json.load(train_sol)
    return eval_cha, eval_sol, train_cha, train_sol

def main():
    data_paths = {
        "train_cha": r"dataset\arc-agi_training_challenges.json",
        "train_sol": r"dataset\arc-agi_training_solutions.json",
        "test_cha": r"dataset\arc-agi_test_challenges.json",
        "eval_sol": r"dataset\arc-agi_evaluation_solutions.json",
        "eval_cha": r"dataset\arc-agi_evaluation_challenges.json"
    }

    eval_cha, eval_sol, train_cha, train_sol = load_data(data_paths)

    print(torch.cuda.is_available())
    problem = random.sample(train_cha.keys(), k=1)[0]

    arc = train_cha[problem]
    sol = train_sol[problem]

    sol = torch.tensor(sol[0])
    model = Y_NET(grid_shape=(1, 30, 30), channels=1, batch_size=1, device='cuda')

    pred, loss_en, loss_de = model.evaluate_on_test(arc_problem=arc, epochs=10, activation=5, en_lr=0.005, de_lr=0.005)
    x_axe = torch.linspace(1, len(loss_en), len(loss_en))

    output = []
    for i in range(pred.shape[0]):
        try:
            output.append(model.grid_scaler(grid=pred[i], mode='down'))
        except Exception as e:
            print(e)

    output = [arc['test'][0]['input']] + output + [sol]

    model.cplot(grid=output, sequence=True, pairs=0, save_path="demo_output.png")

if __name__ == "__main__":
    main()
