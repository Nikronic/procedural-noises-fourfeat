import torch


class OctavePerlin:
    # refs:
    ## 1. http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
    ## 2. https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
    def __init__(self, height=15, width=15, rho=2, n_octaves=4, device=None):
        """
        :param height: Height of grid
        :param width: Width of grid
        :param rho: Base scale for each octave rho^(-i)
        :param n_octaves: Number of octaves to be used
        """

        self.height = height
        self.width = width
        self.device = 'cuda:0' if (torch.cuda.is_available() and device is not None)  else 'cpu'
        self.data = self.populate_grid(rho=rho, n_octaves=n_octaves, device=self.device)

    def __call__(self):
        return self.data

    def populate_grid(self, rho, n_octaves, device=None):
        """
        Computes Perlin Simplex noise over multiple octaves (one octave can be used too)

        :param rho: Base scale for each octave rho^(-i)
        :param n_octaves: Number of octaves to be used
        :return: a ``(height*scale, width*scale)`` Perlin noise with ``len(octaves)`` octaves
        """

        out = self.perlin_ms(octaves=[rho**-i for i in range(n_octaves)], device=device)
        return out
    
    def perlin_ms(self, octaves, device=None):
        """
        Generates a multi-scale Perlin Noise

        :param octaves: A list of magnitude of octaves
        :return: a ``(height*scale, width*scale)`` Perlin noise with ``len(octaves)`` octaves
        """

        height = self.height
        width = self.width

        scale = 2 ** len(octaves)
        out = 0
        for oct in octaves:
            p = self.perlin(width=width, height=height, scale=scale, device=device)
            out += p * oct
            scale //= 2
            width *= 2
            height *= 2
        return out
        
    def perlin(self, width, height, scale, device=None):
        """
        Generates Simplex Perlin noise for given height, width and scale

        :param height: Height of grid
        :param width: Width of grid
        :param scale: Scaler of gridsize
        """
        gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
        xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
        ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)

        wx = 1 - self.fifth_degree_fade(t=xs)
        wy = 1 - self.fifth_degree_fade(t=ys)

        dots = 0
        dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
        dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
        dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
        dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))

        return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)
    
    @staticmethod
    def third_degree_fade(t):
        """
        Fade functions - only continuous in the first-order derivatives

        This eases coordinate values
            so that they will ease towards integral values. This ends up smoothing the final output.

        :param t: coordinates
        :return: Interpolation of coordinates
        """

        return 3 * t**2 - 2 * t ** 3

    @staticmethod
    def fifth_degree_fade(t):
        """
        Fade function - First and second order derivatives are continuous

        This eases coordinate values
            so that they will ease towards integral values. This ends up smoothing the final output.

        :param t: coordinates
        :return: Interpolation of coordinates
        """

        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

## test
# octave_perlin = OctavePerlin(12, 12)
# z1 = octave_perlin(2, 3)
# print(z1.shape)
