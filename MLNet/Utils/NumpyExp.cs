using Numpy;

namespace MLNet.Utils
{
    /// <summary>
    ///     This is a lib extent Numpy
    /// </summary>
    public static class np2
    {
        /// <summary>
        ///     will return NDArray [1,x1,x2,x3,...,xN]
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static NDarray linear_first_order(NDarray a)
        {
            var b = np.ones(a.shape[0]);
            var res = np.insert(a, 0, b, 1);
            return res;
        }

        /// <summary>
        ///     will return NDArray [x1^power,x2^power,x3^power,...,xN^power]
        /// </summary>
        /// <param name="a"></param>
        /// <param name="power"></param>
        /// <returns></returns>
        public static NDarray power(NDarray a, double power)
        {
            var p = np.ones_like(a) * power;
            return np.power(a, p);
        }
    }
}