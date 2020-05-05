
# Pytorch Implementation of Pixel-level Domain Transfer

reference: [Pixel-Level Domain Transfer](https://arxiv.org/pdf/1603.07442.pdf)

## Prerequisites
+ virtualenv
+ PyTorch (tested at 0.3.0, but 0.4.0 would be ok)

## Prepare Dataset
+ Download LOOKBOOK dataset: [Link](https://drive.google.com/file/d/0By_p0y157GxQU1dCRUU4SFNqaTQ/view?usp=sharing)
+ run `sh setup.sh`

## Training 
~~~
python main.py \
  --gpu_id 0 \
  --root_dir 'data/prepro' \
  --csv_file 'data/prepro/label.csv' \
  --expr 'experiment1' \
  --batch_size 24 \
  --load_size 64 \
  --lr 0.0002
~~~

## Visualization
~~~
tensorboard --logdir repo/<expr>/tb --port 8000
~~~

## Bug report
I found the loss dies too quickly. Need to figure out the reason.   
Any pull requests, or bug report is welcome.  :-)

![image1](https://user-images.githubusercontent.com/17468992/41200409-018cfd98-6cdf-11e8-93e7-fc85646c7c89.png)
![image2](https://user-images.githubusercontent.com/17468992/41200410-025498f8-6cdf-11e8-8997-355143a074c4.png)
![image3](https://user-images.githubusercontent.com/17468992/41200411-03075cae-6cdf-11e8-8a10-e14fdda91cc3.png)


## Forked from
MinchulShin / [@nashory](https://github.com/nashory)  
