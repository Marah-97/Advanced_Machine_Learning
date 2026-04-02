import torch
import matplotlib.pyplot as plt

vae1 = {
    "geodesics_distance": [],
    "euclidean_distance": [],
}

vae2 = {
    "geodesics_distance": [],
    "euclidean_distance": [],
}

vae3 = {
    "geodesics_distance": [],
    "euclidean_distance": [],
}

for i in range(10):
    data = torch.load(f"geodesics_data_1_{i}.pt")
    vae1["geodesics_distance"].append(data["geodesics_distance"])
    vae1["euclidean_distance"].append(data["euclidean_distance"])

for i in range(10):
    data = torch.load(f"geodesics_data_2_{i}.pt")
    vae2["geodesics_distance"].append(data["geodesics_distance"])
    vae2["euclidean_distance"].append(data["euclidean_distance"])

for i in range(10):
    data = torch.load(f"geodesics_data_3_{i}.pt")
    vae3["geodesics_distance"].append(data["geodesics_distance"])
    vae3["euclidean_distance"].append(data["euclidean_distance"])


geodesic_distances_vae1 = torch.tensor(vae1["geodesics_distance"])
euclidean_distances_vae1 = torch.tensor(vae1["euclidean_distance"])

geodesic_distances_vae2 = torch.tensor(vae2["geodesics_distance"])
euclidean_distances_vae2 = torch.tensor(vae2["euclidean_distance"])

geodesic_distances_vae3 = torch.tensor(vae3["geodesics_distance"])
euclidean_distances_vae3 = torch.tensor(vae3["euclidean_distance"])


geo_CoV_vae1  = torch.var(geodesic_distances_vae1) / torch.mean(geodesic_distances_vae1)
euclidean_CoV_vae1 = torch.var(euclidean_distances_vae1) / torch.mean(euclidean_distances_vae1)

geo_CoV_vae2  = torch.var(geodesic_distances_vae2) / torch.mean(geodesic_distances_vae2)
euclidean_CoV_vae2 = torch.var(euclidean_distances_vae2) / torch.mean(euclidean_distances_vae2)

geo_CoV_vae3  = torch.var(geodesic_distances_vae3) / torch.mean(geodesic_distances_vae3)
euclidean_CoV_vae3 = torch.var(euclidean_distances_vae3) / torch.mean(euclidean_distances_vae3)

print(f"VAE 1 - Geodesic CoV: {geo_CoV_vae1:.4f}, Euclidean CoV: {euclidean_CoV_vae1:.4f}")
print(f"VAE 2 - Geodesic CoV: {geo_CoV_vae2:.4f}, Euclidean CoV: {euclidean_CoV_vae2:.4f}")
print(f"VAE 3 - Geodesic CoV: {geo_CoV_vae3:.4f}, Euclidean CoV: {euclidean_CoV_vae3:.4f}")


[1,1,2,2,3,3]
[geo_CoV_vae1, geo_CoV_vae2, geo_CoV_vae3]
[euclidean_CoV_vae1, euclidean_CoV_vae2, euclidean_CoV_vae3]

plt.plot([1,1,2,2,3,3], [geo_CoV_vae1, geo_CoV_vae1, geo_CoV_vae2, geo_CoV_vae2, geo_CoV_vae3, geo_CoV_vae3], 'bo-', label='Geodesic CoV')
plt.plot([1,1,2,2,3,3], [euclidean_CoV_vae1, euclidean_CoV_vae1, euclidean_CoV_vae2, euclidean_CoV_vae2, euclidean_CoV_vae3, euclidean_CoV_vae3], 'ro-', label='Euclidean CoV')
plt.xlabel('Number of Decoders')
plt.ylabel('Coefficient of Variation (CoV)')
plt.title('CoV of Geodesic and Euclidean Distances for Different VAEs')
plt.xticks([1, 2, 3])
plt.legend()
plt.grid()
plt.show()



    

