//Author: Ugo Varetto
//creates a number of threads then calls ps uH p from within the program
//and dumps the output; useful when the ps program is not directly accessible
//as with e.g. aprun on Cray.
//
//compile with -std=c++ and -pthread flags
//If you want to compile with CC on Cray XK-7 anc XC-30 
//do load the craype-accel-* modules, if not you'll get a segfault
//
//note that the total number of threads is the number passed on the command
//line + 1 (the first thread created in the process)

#if __cplusplus < 201103L
#error "C++ 11 compiler required"
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <algorithm>
#include <iterator>
#include <unistd.h>


//------------------------------------------------------------------------------
std::string threadreport() {
    const pid_t pid = getpid();
    std::ostringstream oss;
    oss << "ps uH p " << pid;
    FILE* f = popen(oss.str().c_str(), "r");
    std::string psout;
    std::vector< char > buffer(0x10000);
    size_t count = fread(&buffer[0], sizeof(char), buffer.size(), f);
    while(count) {
            buffer[count] = '\0';
            psout += &buffer[0];
            count = fread(&buffer[0], sizeof(char), buffer.size(), f);
    }
    pclose(f);
    return psout;
}

typedef std::vector< std::future< void > > Futures;

Futures createthreads(bool& wait, int num) {
    Futures futures;
    for(int i = 0; i != num; ++i) 
        futures.push_back(std::async(std::launch::async, [&wait]{
            while(wait)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }));
    return futures;
}

template < typename FI >
void barrier(FI begin, FI end) {
    typedef typename std::iterator_traits< FI >::value_type FutureType;
    std::for_each(begin, end, [](FutureType& t) {
                                t.wait();
                              });
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {
    const int numthreads = argc > 1 ? atoi(argv[1]) : 0;
    bool wait = true;
    Futures futures(createthreads(wait, numthreads));
    std::cout << threadreport() << std::endl;
    wait = false;    
    barrier(futures.begin(), futures.end());
    return 0;
}
