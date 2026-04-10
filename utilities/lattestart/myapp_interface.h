#ifndef __MYAPP_INTERFACE_H
#define __MYAPP_INTERFACE_H

class MyAppInterface
{
public:
    MyAppInterface() {}
    virtual ~MyAppInterface() {}

    virtual bool OnInit() = 0;

private:
    MyAppInterface( const MyAppInterface& source );
    void operator = ( const MyAppInterface& source );
};


#endif // __MYAPP_INTERFACE_H
