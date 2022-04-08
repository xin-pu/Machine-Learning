namespace MLNet.Utils
{
    public class EnumerableExt
    {
        public static double[] GetList(double start, double end, int length = 1)
        {
            var step = (start - end) / length;
            var list = Enumerable.Range(0, length).Select(i => start + i * step).ToArray();
            return list;
        }

        public static double[,] GetList(double[] array)
        {
            var dims = array.GetLength(0);
            var res = new double[dims, 1];
            Enumerable.Range(0, dims).ToList().ForEach(i => { res[i, 0] = array[i]; });
            return res;
        }
    }
}