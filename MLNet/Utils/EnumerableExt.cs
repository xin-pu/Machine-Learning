namespace MLNet.Utils
{
    public class EnumerableExt
    {
        public static double[] GetLinearArray(double start, double end, int length = 1)
        {
            var step = (end - start) / length;
            var list = Enumerable.Range(0, length).Select(i => start + i * step).ToArray();
            return list;
        }

        /// <summary>
        ///     Convert double[x] to double[x,1]
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double[,] GetDyadicArray(double[] array)
        {
            var dims = array.GetLength(0);
            var res = new double[dims, 1];
            Enumerable.Range(0, dims).ToList().ForEach(i => { res[i, 0] = array[i]; });
            return res;
        }

        /// <summary>
        ///     Convert [start ... end] to double[x,1]
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static double[,] GetDyadicArray(double start, double end, int length = 1)
        {
            var array = GetLinearArray(start, end, length);
            return GetDyadicArray(array);
        }
    }
}