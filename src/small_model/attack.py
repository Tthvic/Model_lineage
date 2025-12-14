import torch
import torch.nn as nn
import numpy as np

class MinADAttack:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Attack parameters from config
        self.num_attack_iter = config['generation']['adversarial'].get('iters', 10)
        self.attack_lr_begin = config['generation']['adversarial'].get('lr_begin', 0.1)
        self.attack_lr_end = config['generation']['adversarial'].get('lr_end', 0.001)
        self.num_evals_boundary = config['generation']['adversarial'].get('num_evals', 100)
        self.epsilon = config['generation']['adversarial'].get('epsilon', 0.03)

    def loss_cos(self, x_adv, x_ori, adv_label, distance):
        x_sub = x_adv - x_ori
        x_sub = x_sub.view(-1)
        x_sub = x_sub / torch.norm(x_sub, p=2)

        grad_xadv = self.gradient_compute_boundary(x_adv, adv_label, distance)
        grad_xadv = grad_xadv.view(-1)
        grad_xadv = grad_xadv / torch.norm(grad_xadv, p=2)

        cos = -torch.cosine_similarity(x_sub, grad_xadv, dim=-1)
        return cos

    def gradient_compute_boundary(self, sample, adv_label, distance):
        sample = sample.squeeze(0)
        num_evals = self.num_evals_boundary
        
        delta = 1 / len(sample.shape) * distance
        if delta > 0.2:
            delta = 0.2

        noise_shape = [num_evals] + list(sample.shape)
        rv = torch.randn(*noise_shape, device=self.device)
        rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=(1, 2, 3), keepdim=True))

        perturbed = sample + delta * rv
        decisions = self.decision_function(perturbed, adv_label)
        decision_shape = [len(decisions)] + [1] * len(sample.shape)
        fval = 2 * decisions.reshape(decision_shape) - 1.0

        if torch.mean(fval) == 1.0:
            gradf = torch.mean(rv, dim=0)
        elif torch.mean(fval) == -1.0:
            gradf = -torch.mean(rv, dim=0)
        else:
            fval -= torch.mean(fval)
            gradf = torch.mean(fval * rv, dim=0)

        return gradf

    def binary_search(self, x_0, x_random, adv_label, tol=1e-2):
        adv = x_random.detach().clone()
        cln = x_0.detach().clone()
        
        adv = adv.to(self.device)
        cln = cln.to(self.device)

        while True:
            mid = ((cln + adv) / 2.0).to(self.device)
            if self.decision_function(mid, adv_label):
                adv = mid
            else:
                cln = mid
            
            if torch.norm(adv - cln) < tol:
                break
        return adv.detach()

    def decision_function(self, images, adv_label, batch_size=100):
        self.model.eval()
        
        if len(images.shape) < 4:
            with torch.no_grad():
                predict_label = torch.argmax(self.model(images.unsqueeze(0)), dim=1)
                predict_label = predict_label.reshape(1, 1)
        else:
            results_list = []
            num_batch = int(np.ceil(len(images) / float(batch_size)))
            for m in range(num_batch):
                begin = m * batch_size
                end = min((m + 1) * batch_size, images.shape[0])
                
                with torch.no_grad():
                    output = self.model(images[begin:end])
                output = output.detach().cpu().numpy().astype(np.float32)
                results_list.append(output)

            results = np.vstack(results_list)
            predict = torch.from_numpy(results).to(self.device)
            predict_label = torch.argmax(predict, dim=1).reshape(len(images), 1)

        target_label = torch.zeros((len(images), 1), device=self.device)
        target_label[:] = adv_label
        return predict_label == target_label

    def adv_break_flag(self, image, ori_label, tar_label, threshold=0.1):
        with torch.no_grad():
            output = self.model(image)
        top2_values, top2_indices = torch.topk(output, k=2, dim=1)
        top2_indices = top2_indices.squeeze().tolist()
        
        if set(top2_indices) == {ori_label.item(), tar_label.item()}:
            value_diff = abs(top2_values[0, 0].item() - top2_values[0, 1].item())
            if value_diff < threshold:
                return True
        return False

    def attack(self, x_sample, y_sample, tgt_sample, adv_label):
        # Step 1: Initial adversarial sample at boundary
        adv_init = self.binary_search(x_sample, tgt_sample, adv_label)
        adv_update = adv_init.detach().clone()
        ori_sample = x_sample.detach().clone()

        distance_init = torch.norm(adv_init - ori_sample, p=2)
        distance_value = distance_init

        lr1 = self.attack_lr_begin
        temp_list = []

        for attack_epoch in range(self.num_attack_iter):
            adv_sample = adv_update.detach().clone()
            distance = distance_value.detach().clone()

            adv_sample.requires_grad = True
            loss = self.loss_cos(adv_sample, ori_sample, adv_label, distance)
            loss.backward()
            
            if adv_sample.grad is None:
                grads = torch.zeros_like(adv_sample)
            else:
                grads = adv_sample.grad / (torch.norm(adv_sample.grad, p=2) + 1e-8)
            
            adv_sample.requires_grad = False

            lr2 = lr1
            temp = 0
            adv_sample -= lr1 * grads

            if len(adv_sample.shape) < 4:
                adv_sample = adv_sample.unsqueeze(0)
            
            while not self.decision_function(adv_sample, adv_label):
                temp = 1
                lr2 /= 2
                adv_sample += lr2 * grads
                if lr2 < self.attack_lr_end / 2:
                    break

            temp_list.append(temp)

            if not temp and lr1 > self.attack_lr_end or sum(temp_list[max(0, attack_epoch - 4):attack_epoch]) == 4:
                lr1 /= 2
                if lr1 < self.attack_lr_end:
                    lr1 = self.attack_lr_end

            with torch.no_grad():
                adv_sample = self.binary_search(ori_sample, adv_sample, adv_label)
                distance_value = torch.norm(adv_sample - ori_sample, p=2)

            adv_update = adv_sample.detach().clone()

            if self.adv_break_flag(adv_sample, y_sample, adv_label):
                break

        return adv_update.detach().clone(), distance_value.item()
