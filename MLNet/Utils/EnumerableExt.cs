namespace MLNet.Utils;

public class EnumerableExt
{
    public static List<double> GetList(double start, double end, int length = 1)
    {
        var step = (start - end) / length;
        var list = Enumerable.Range(0, length).Select(i => start + i * step).ToList();
        return list;
    }
}