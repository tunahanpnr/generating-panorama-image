from panoramic import Panoramic

if __name__ == '__main__':
    img_paths = [
        './images/1.jpg',
        './images/2.jpg',
        './images/3.jpg'
    ]
    level = 4

    panoramic = Panoramic(img_paths, level)
    panoramic.run()
    panoramic.save()
