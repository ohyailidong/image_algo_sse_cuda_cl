#ifndef COMMON_DATA_DEFINE_H
#define COMMON_DATA_DEFINE_H

struct Point {
	int x;
	int y;
	Point(int ix, int iy) { x = ix; y = iy; }
	~Point(){}
};
struct Image
{
	int width;
	int height;
	int channel;
	void* data;
	Image() :width(0), height(0), channel(0), data(nullptr) {}
	Image(int iwidth, int iheight, int ichannel, void* p) :width(iwidth),
		height(iheight),
		channel(ichannel),
		data(p) {}
	~Image(){}
	int GetElementNum() { return width * height* channel; }
};

enum BorderTypes {
	BORDER_CONSTANT = 0, //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
	BORDER_REPLICATE = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
	BORDER_REFLECT = 2, //!< `fedcba|abcdefgh|hgfedcb`
	BORDER_WRAP = 3, //!< `cdefgh|abcdefgh|abcdefg`
	BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
	BORDER_TRANSPARENT = 5, //!< `uvwxyz|abcdefgh|ijklmno`

	BORDER_REFLECT101 = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
	BORDER_DEFAULT = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
	BORDER_ISOLATED = 16 //!< do not look outside of ROI
};
enum
{
	TM_SQDIFF = 0,
	TM_SQDIFF_NORMED = 1,
	TM_CCORR = 2,
	TM_CCORR_NORMED = 3,
	TM_CCOEFF = 4,
	TM_CCOEFF_NORMED = 5
};
enum MorphTypes {
	MORPH_ERODE = 0, 
	MORPH_DILATE = 1, 
	MORPH_OPEN = 2, 
	MORPH_CLOSE = 3, 
	MORPH_GRADIENT = 4, //!< a morphological gradient
	MORPH_TOPHAT = 5, //!< "top hat"
	MORPH_BLACKHAT = 6, //!< "black hat"
	MORPH_HITMISS = 7  //!< "hit or miss"
};
//! shape of the structuring element
enum MorphShapes {
	MORPH_RECT = 0, 
	MORPH_CROSS = 1, 
	MORPH_ELLIPSE = 2 
};
#endif
