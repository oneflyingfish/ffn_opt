import torch
import torch.nn as nn

def create_ffn_list(num_blocks, dim, ffn_dim):
    layers = nn.ModuleList()
    for _ in range(num_blocks):
        ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), 
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )
        layers.append(ffn)
    return layers

dim = 5120
ffn_dim = 13824
num_blocks = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_ffns = create_ffn_list(num_blocks, dim, ffn_dim).to(device)

save_path = "/mnt/shanhai-ai/shanhai-workspace/zhouyang/infra_triton/wan_high_noise_ffn_modulelist.pt"
state_dict = torch.load(save_path, map_location=device)
my_ffns.load_state_dict(state_dict)
my_ffns.eval()

x = torch.randn(1, 128, dim).to(device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


with torch.no_grad():
    start_event.record()
    for i in range(num_blocks):
        x = my_ffns[i](x)
    end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)

print(f"{elapsed_time:.3f} ms")
print(f"{x.shape}")