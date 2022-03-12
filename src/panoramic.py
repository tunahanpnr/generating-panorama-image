import cv2
import numpy as np


def prep_flann():
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


class Panoramic:
    def __init__(self, img_paths, level=4):
        try:
            self.images = [cv2.imread(path) for path in img_paths]
            self.images[1] = cv2.copyMakeBorder(self.images[1], 200, 200, 500, 500, cv2.BORDER_CONSTANT)
            self.SIFT = []
            self.ransac = []
            self.result = None
            self.flann = prep_flann()
            self.level = level
        except:
            print('Please make sure your image paths are correct!')
            return

    # extract the sift features
    def extract_SIFT_features(self):
        for img in self.images:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)
            self.SIFT.append([kp, des])

    # match the extracted features via flann
    def match_features(self, other, middle):
        dest2src = self.flann.knnMatch(self.SIFT[middle][1], self.SIFT[other][1], k=2)

        ratio = [[0, 0] for i in range(len(dest2src))]
        matches = {}
        for i, (m, n) in enumerate(dest2src):
            if m.distance < 0.7 * n.distance:
                ratio[i] = [1, 0]
                matches[m.trainIdx] = m.queryIdx

        good = []
        src2dest = self.flann.knnMatch(self.SIFT[other][1], self.SIFT[middle][1], k=2)
        ratio = [[0, 0] for i in range(len(src2dest))]

        for i, (m, n) in enumerate(src2dest):
            if m.distance < 0.7 * n.distance:
                if m.queryIdx in matches and matches[m.queryIdx] == m.trainIdx:
                    good.append(m)
                    ratio[i] = [1, 0]
        matched_1 = [self.SIFT[other][0][m.queryIdx].pt for m in good]
        matched_2 = [self.SIFT[middle][0][m.trainIdx].pt for m in good]
        self.ransac_and_homography(matched_1, matched_2)

    # homography with RANSAC
    def ransac_and_homography(self, src, dest):
        src_pts = np.float32(src).reshape(-1, 1, 2)
        dst_pts = np.float32(dest).reshape(-1, 1, 2)

        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        self.ransac.append([m, src, dest, mask])

    def warp_perspective(self):
        ransac_src = self.ransac[0]
        ransac_dest = self.ransac[1]

        ones_left = np.ones_like(self.images[0], dtype='float32')
        ones_right = np.ones_like(self.images[2], dtype='float32')

        out1 = cv2.warpPerspective(self.images[0], ransac_src[0], (self.images[1].shape[1], self.images[1].shape[0]))
        out2 = cv2.warpPerspective(self.images[2], ransac_dest[0], (self.images[1].shape[1], self.images[1].shape[0]))
        out3 = cv2.warpPerspective(ones_left, ransac_src[0], (self.images[1].shape[1], self.images[1].shape[0]))
        out4 = cv2.warpPerspective(ones_right, ransac_dest[0], (self.images[1].shape[1], self.images[1].shape[0]))

        blended = self.blend(out1, self.images[1], out3)
        self.result = self.blend(out2, blended, out4)

    def blend(self, pers1, pers2, pers3):

        lp_pyr_1 = self.create_pyramid(pers1, False)
        lp_pyr_2 = self.create_pyramid(pers2, False)
        lp_pyr_3 = self.create_pyramid(pers3, True)

        blended = []
        for l1, l2, gm in zip(lp_pyr_1, lp_pyr_2, lp_pyr_3):
            ls = l1 * gm + l2 * (1.0 - gm)
            blended.append(ls)

        # now reconstruct
        reconstructed = blended[0]
        for i in range(1, self.level):
            reconstructed = cv2.pyrUp(reconstructed)
            reconstructed = cv2.add(reconstructed, blended[i])

        return reconstructed

    def create_pyramid(self, img, reverse):
        p = img.copy()
        pyr = [p]
        for i in range(self.level):
            p = cv2.pyrDown(p)
            pyr.append(np.float32(p))

        lp_pyr = [pyr[self.level - 1]]
        for i in range(self.level - 1, 0, -1):
            if not reverse:
                lap = np.subtract(pyr[i - 1], cv2.pyrUp(pyr[i]))
                lp_pyr.append(lap)
            else:
                lp_pyr.append(pyr[i - 1])

        return lp_pyr

    def run(self):
        self.extract_SIFT_features()
        self.match_features(0, 1)
        self.match_features(2, 1)
        self.warp_perspective()

    def save(self):
        if self.result is not None:
            cv2.imwrite(f'panoramic_output_test_l{self.level}.png', self.result)
        else:
            print('Please first execute run function!')
