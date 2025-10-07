Main problem : I want to use cython for calculating convolution fast
I already implemented back in a while with 2 for loops, but it was slow and not fast at all !
So for now I used scipy one : 
    def apply_kernel(self, image, kernel):
        flipped_kernel = rotate_kernel(kernel)
        output = convolve2d(
            image, flipped_kernel, mode="same", boundary="fill", fillvalue=0
        )
        return output

But I am not satisfied with such a core funtionality of my library to be a dependency !


