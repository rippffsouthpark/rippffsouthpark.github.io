const SHELL_CACHE = "qwen-shell-v3";

const SHELL_FILES = [
    "./",
    "./index.html",
    "./app.js",
    "./manifest.json"
];


// ------------------------------------------------------------
// INSTALL
// ------------------------------------------------------------

self.addEventListener(
    "install",
    event => {

        event.waitUntil(

            caches.open(
                SHELL_CACHE
            ).then(
                async cache => {

                    // Cache files one-by-one.
                    // If one fails, the other files still work.

                    for (
                        const file
                        of SHELL_FILES
                    ) {

                        try {

                            await cache.add(
                                file
                            );

                        } catch (error) {

                            console.warn(
                                "Could not cache:",
                                file,
                                error
                            );
                        }
                    }
                }
            )
        );

        self.skipWaiting();
    }
);


// ------------------------------------------------------------
// ACTIVATE
// ------------------------------------------------------------

self.addEventListener(
    "activate",
    event => {

        event.waitUntil(

            caches.keys().then(
                keys =>

                    Promise.all(

                        keys
                            .filter(
                                key =>
                                    key !==
                                    SHELL_CACHE
                            )
                            .map(
                                key =>
                                    caches.delete(
                                        key
                                    )
                            )
                    )
            )
        );

        self.clients.claim();
    }
);


// ------------------------------------------------------------
// FETCH
// ------------------------------------------------------------

self.addEventListener(
    "fetch",
    event => {

        if (
            event.request.method !==
            "GET"
        ) {

            return;
        }


        const url =
            new URL(
                event.request.url
            );


        // Only intercept files belonging
        // to this GitHub Pages site.

        if (
            url.origin !==
            self.location.origin
        ) {

            return;
        }


        event.respondWith(

            caches.match(
                event.request
            ).then(
                cached => {

                    if (cached) {

                        return cached;
                    }


                    return fetch(
                        event.request
                    ).then(
                        response => {

                            if (
                                response.ok
                            ) {

                                const copy =
                                    response.clone();

                                caches.open(
                                    SHELL_CACHE
                                ).then(
                                    cache =>
                                        cache.put(
                                            event.request,
                                            copy
                                        )
                                );
                            }


                            return response;
                        }
                    );
                }
            )
        );
    }
);