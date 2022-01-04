# RadiationPatternRebuild
Restoring antenna's radiation pattern using GAN model  
This is just a inital trial that checks if GAN is suitable for the topic  
- Hiding the data and results  
- Hiding cGAN model due to confidentiality  

# Instruction
- "config.py": Set configurations  
- "models.py": All NN models are here  
- "numpyData_import.py": 
    Import data  
    Array with the size of (64, 64)  
    Each element is the measurement at point (theta, phi) at far field  
- "main.py": Executing the model and exporting the result


