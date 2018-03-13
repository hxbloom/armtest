#include <time.h>
#include <iostream>

using namespace std;

void foo()
{
//    int m = 0;
//    for(int i=0; i<30; )
//    {
//        i++;
//    }
};


int main (int argc, char** argv)
{
    // reset the clock
    timespec tS;
    tS.tv_sec = 0;
    tS.tv_nsec = 0;
    clock_settime(CLOCK_PROCESS_CPUTIME_ID, &tS);

    //foo();
    //foo();
    //foo();
    //foo();

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tS);
    cout << "Time taken is: " << tS.tv_sec << "+" << tS.tv_nsec /1000000000.0 << " seconds. " << endl;

    return 0;
}
