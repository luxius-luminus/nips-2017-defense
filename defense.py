import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
import csv
import PIL
import os
import glob
from torch.autograd import Variable
from transformer import Transformer4 as Transformer
parser = argparse.ArgumentParser()

parser.add_argument('-cp1','--checkpoint_path1' ,help='Path to checkpoint for inception network1.')

parser.add_argument('-cp2','--checkpoint_path2' ,help='Path to checkpoint for inception network2.')
    
parser.add_argument('-i','--input_dir',help = 'Input directory with images.')

parser.add_argument('-o','--output_file',help = 'Output flie to save labels.')

parser.add_argument('-w','--image_width',default = 299, type = int, help = 'Width of each input images.')

parser.add_argument('-ih','--image_height',default = 299,type = int, help = 'Height of each input images.')

parser.add_argument('-b','--batch_size',default = 16, type = int, help = 'How many images process at one time.')

def load_images(input_dir,batch_shape,transform=None):
    
    images = torch.zeros(batch_shape)
    filenames=[]
    idx = 0
    batch_size = batch_shape[0]
    for filename in glob.glob(os.path.join(input_dir, '*.png')):
        image = PIL.Image.open(filename)
        if transform is not None:
            image = transform(image)
        images[idx,:,:,:] = image
        filenames.append(os.path.basename(filename))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = torch.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images
            
            
def main():
    args = parser.parse_args()
    batch_shape = [args.batch_size, 3, args.image_height, args.image_width]
    use_gpu = torch.cuda.is_available()
    # initialize model
    model1 = models.inception_v3(pretrained=False, transform_input=True)
    model2 = models.inception_v3(pretrained=False, transform_input=True)
    transformer = Transformer() 
    
    # load model parameters
    checkpoint1 = torch.load(args.checkpoint_path1)
    stateDict1 = checkpoint1['stateDict']
    model1.load_state_dict(stateDict1)
    if use_gpu:
        model1.cuda()
    model1.eval()
    
    checkpoint2 = torch.load(args.checkpoint_path2)
    stateDict2 = checkpoint2['stateDict']
    model2.load_state_dict(stateDict2)
    if use_gpu:
        model2.cuda()
    model2.eval()
    
    checkpoint3 = torch.load('./tar.tar')
    stateDict3 = checkpoint3['state_dict']
    transformer.load_state_dict(stateDict3)
    if use_gpu:
        transformer.cuda()
    transformer.eval()


    # set transformation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Scale(299),
                                    transforms.CenterCrop(299),
                                    transforms.ToTensor(),
                                    normalize,
                                   ])
    
    # run classfication
    for filenames, images in load_images(args.input_dir, batch_shape,transform):
        if use_gpu:
            images = Variable(images.cuda(),volatile=True)
        else:
            images = Variable(images,volatile=True)

        images = transformer(images)

        outputs1 = model1(images)
        outputs2 = model2(images)
        outputs=outputs1+outputs2

        _, labels = torch.max(outputs.data, 1)
        labels = labels +1
        csvfile = open(args.output_file,'a')
        writer = csv.writer(csvfile)
        for filename, label in zip(filenames, labels):
            writer.writerow([filename,label])
        csvfile.close()
        
        
        


if __name__ == '__main__':
    main()
    
