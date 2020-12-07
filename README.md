# Story Maker

Story Maker is a website which can generate personalized character based on speech input. The application is designed for kids who are too little to draw or express their own character precisely. Users can set their constraints for eyes color, hair color, skin color, and whether wearing glasses through speech input. Then the website would randomly generate 6 figures that meet the giving constraints using ACGAN. The website support Chinese only.


* Training Dataset: 

    [Cartoon set](https://google.github.io/cartoonset/)
    
    ![](https://i.imgur.com/JNQ6Qtm.png)

* Features:
    | Hair colors | Glasses | Eyes colors | Face colors |
    | ------- | -------- | ------ | ------ |
    | Blue    | With     | Blue   | Yellow |
    | White   | without  | Green  | Black  |
    | Orange  |          | Black  | White  |
    | Brown   |          | Brown  |        |
    | Black   |          |        |        |
    | Yellow  |          |        |        |


## Demo Website
Please click the link to see the temporary website deployed on GCP

URL: https://35.232.8.145:8000/

Due day: 2021-02-26



## Getting Started

### Step 1: Download pretrained model

1. Download the GAN model through the link: [download link](https://www.dropbox.com/s/pzghwmcg3l3fpkf/best.ckpt?dl=0)
2. Put the file under the folder "demowebsite"

### Step 2: GCP Cloud Speech-to-Text API

1. Register for GCP Cloud Speech-to-Text API
2. Download the google application credential file and put it under the folder "demowebsite"

### Step 3: Install package

1. Install Pytorch from: https://pytorch.org/
2. Install ffmpeg
3. Run 
   ```python3
    sudo apt-get install libsndfile1
    pip3 install -r requirement.txt
   ```

## Screenshots
1. **Homepage: Click on the button**

    ![](https://i.imgur.com/zmDpk7g.png)
    
2. **Click on the left button to start recording. Then click on the right one to stop.**
   Sample Input: 「我想要我的角色有橘色的頭髮，然後我喜歡亞洲人」("I would like my character with orange hair. And I prefer Asian people."")

    ![](https://i.imgur.com/VNamtdm.png)
    
3. **Generating figures**

    ![](https://i.imgur.com/h4wFN5q.png)

4. **Receive 6 pictures**

    ![](https://i.imgur.com/joUc5nG.png)





