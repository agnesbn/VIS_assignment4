# Assignment 4 – Human Action classification
The portfolio for __Visual Analytics S22__ consists of 4 projects (3 class assignments and 1 self-assigned project). This is the __final assignment__ in the portfolio.


## 1. Contribution
..

## 2. Assignment description
...

## 3. Methods


## 4. Usage
Put the flower-data into the `in` folder.

Before running the script, run the following in the Terminal:
```
pip install --upgrade pip
pip install opencv-python
sudo apt-get update
sudo apt-get -y install graphviz
```
Then, from the `VIS_assignment1` directory, run:
```
python src/image_search_hist.py --image_index {INDEX}
```
`{INDEX}` represents an user-defined argument. Here, you can write any number from 0–1359 and it will index your target image.

## 5. Discussion of results
When I ran the code, I put in `231` as my target image index. The results can be seen in the `out` folder. The output files are:
- `hist_similar_images_indx231.csv`: A CSV with a row for the name of the target image, 
- `hist_similar_images_indx231.png`: An image of the target image and its three most similar images with their respective distance scores.

As you can tell if you look at the output, the method was relatively sucessful in my case. The flowers that were identified as the most similar to my target image do indeed seem to be of the same species as my target flower. I did find, however, that it did not work as well with all images, as it did for image_0232. If you put in `600`, for example, it seems as though none of the flowers on the three most similar images are the same as the one on the target image. And what's more, they are not even the same flowers between them. It higlights the weakness of this type of method. The fact that it only focusses on colour ...

Thus, for this type of task a more advanced method would be beneficiary. E.g. using a pretrained model, like VGG16, to do feature extraction for all images and then using a nearest neighbour algorithm to find the images with less "distance".
