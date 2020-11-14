import torch 

a = torch.tensor([[1,2,3],[4,5,6]])
print(f"1. shape of 'a': H x W =  {a.shape[0]} x {a.shape[1]} \n",a)
b = torch.stack((a,a),dim=-1)
print("2. stack 'a' twice to get 'b' \n")
print(f"shape of 'b': H x W x C=  {b.shape[0]} x {b.shape[1]} x {b.shape[2]} \n",b)
print("3. reshpe 'b' to get 'c' \n")
c = b.reshape(b.shape[0],b.shape[1]*b.shape[2])
print(f"shape of 'c': H x (WC) = {c.shape[0]} x {c.shape[1]}\n")
print(c)