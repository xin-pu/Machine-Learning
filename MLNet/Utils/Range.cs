namespace MLNet.Utils
{
    public class range
    {
        /// <summary>
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public static double[] linspace(double start, double end, int length = 1)
        {
            var step = (end - start) / length;
            var list = Enumerable.Range(0, length).Select(i => start + i * step).ToArray();
            return list;
        }

        /// <summary>
        ///     Convert [start ... end] to double[x,1]
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double[,] matrix(double start, double end, int length = 1)
        {
            var array = linspace(start, end, length);
            return matrix(array);
        }


        /// <summary>
        ///     Convert double[x] to double[x,1]
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double[,] matrix(double[] array)
        {
            var dims = array.GetLength(0);
            var res = new double[dims, 1];
            Enumerable.Range(0, dims).ToList().ForEach(i =>
                res[i, 0] = array[i]);
            return res;
        }
    }
}