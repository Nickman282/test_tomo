


            




                    

# 'Reconstruction Diameter' 'Rescale Intercept' 'Rescale Slope'
        



with open(os.path.join(os.getcwd(), "settings.json")) as f:
    param_dict = json.load(f)

filepaths = param_dict["filepaths"]

processor = Processor()
print(processor.sample_loadin(filepaths[0]).columns.values)


