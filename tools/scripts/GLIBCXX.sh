strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
strings /DDN_ROOT/jfxiao/conda_envs/mae/lib/libstdc++.so.6.0.29 | grep GLIBCXX
cp /DDN_ROOT/jfxiao/conda_envs/mae/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/
rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
