# Student-Friendly Multi-Teacher Knowledge Distillation for Point Cloud-based 3D Object Detection 

High-performance LiDAR-based 3D object detectors often face significant computational overhead, necessitating their compression into lightweight detectors. Knowledge distillation (KD) is an effective approach for sparse 3D object detection compression.  

We propose **Student-Friendly Multi-Teacher Knowledge Distillation for Point Cloud-based 3D Object Detection （SFMKD）**, which:  
- Generates **student-friendly teacher assistants** through structured and unstructured pruning.  
- Introduces a **Adaptive Softmax-T-based Multi-Stage Strategy**, incorporating a **multi-stage Softmax-T strategy** for adaptive weighting.  
- Proposes **Multi-Teacher Channel Alignment Layer** and **Multi-Teacher Shared Masks** to reduce the difficulty of knowledge transfer.  

Extensive experiments on **Waymo & KITTI & NuScense datasets** demonstrate that our method achieves a **1.8-4.0× reduction in FLOPs and parameters** without noticeable accuracy loss in LiDAR-based 3D object detection.  


