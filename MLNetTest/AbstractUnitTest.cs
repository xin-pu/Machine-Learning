using MathNet.Numerics.Random;
using Xunit.Abstractions;

namespace MLNetTest;

public abstract class AbstractUnitTest
{
    protected AbstractUnitTest(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
        _testOutputHelper.WriteLine(ToString());
    }

    internal SystemRandomSource SystemRandomSource => SystemRandomSource.Default;
    internal ITestOutputHelper _testOutputHelper { get; }

    internal void Print<T>(T[] array)
    {
        _testOutputHelper.WriteLine(string.Join($",", array));
    }
}