package main

import (
	"fmt"
	"net/http"
	"os"
	"time"
)

// Probe checks that the queue gateway answers on /healthz within a deadline.
type Probe struct {
	URL     string
	Timeout time.Duration
}

// Check returns nil when the gateway responds with HTTP 200.
func (p Probe) Check() error {
	client := http.Client{Timeout: p.Timeout}
	resp, err := client.Get(p.URL + "/healthz")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unhealthy: status %d", resp.StatusCode)
	}
	return nil
}

func main() {
	probe := Probe{URL: os.Getenv("QUEUE_URL"), Timeout: 2 * time.Second}
	if err := probe.Check(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("ok")
}
